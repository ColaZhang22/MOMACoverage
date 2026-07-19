"""Self-contained PCMA implementation for continuous-action BenchMARL tasks.

This module implements a practical adaptation of Preference Coordinated
Multi-agent Policy Optimization (PCMA):

    Wang et al., "Learning Coordinated Preference for Multi-Objective
    Multi-Agent Reinforcement Learning", 2026.

It deliberately does not register itself in :mod:`benchmarl.algorithms` and
does not require a Hydra configuration file.  Run it directly with:

    python -m benchmarl.algorithms.pcma --env-path PATH_TO_UNITY_ENV

The current SAR environments expose a global preference, a scalarised reward,
and an agent-wise vector reward.  PCMA preserves the global preference as the
user request, learns a Dirichlet-distributed local preference for every agent,
and uses the mean agent vector reward as its team vector reward.  This is the
closest no-environment-change adaptation of the paper's separate team and
individual reward streams.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Dirichlet, Normal
from torchrl.envs.utils import ExplorationType, exploration_type
from torchrl.modules import TanhNormal

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    device,
) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend((nn.Linear(last_dim, hidden_dim), nn.Tanh()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers).to(device)


def _agent_embedding(
    embedding: nn.Embedding, reference: torch.Tensor, n_agents: int
) -> torch.Tensor:
    ids = torch.arange(n_agents, device=reference.device)
    encoded = embedding(ids)
    return encoded.view(
        *((1,) * (reference.ndim - 2)), n_agents, encoded.shape[-1]
    ).expand(*reference.shape[:-2], n_agents, encoded.shape[-1])


class PreferencePlanner(nn.Module):
    """Shared Dirichlet planner producing one local preference per agent."""

    def __init__(
        self,
        observation_dim: int,
        preference_dim: int,
        n_agents: int,
        hidden_sizes: Sequence[int],
        agent_embedding_dim: int,
        alpha_min: float,
        alpha_max: float,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.preference_dim = preference_dim
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if preference_dim != 2:
            raise ValueError(
                "The SAR PCMA planner expects two objectives: [p0, p1]"
            )
        self.register_buffer(
            "_active_preference_indices",
            torch.tensor((0, 1), dtype=torch.long, device=device),
            persistent=False,
        )
        self.agent_id = nn.Embedding(n_agents, agent_embedding_dim).to(device)
        self.network = _mlp(
            observation_dim + preference_dim + agent_embedding_dim,
            hidden_sizes,
            2,
            device=device,
        )

    def forward(
        self, observation: torch.Tensor, global_preference: torch.Tensor
    ) -> torch.Tensor:
        ids = _agent_embedding(self.agent_id, observation, self.n_agents)
        inputs = torch.cat(
            (
                observation.to(torch.float32),
                global_preference.to(torch.float32),
                ids,
            ),
            dim=-1,
        )
        alpha = F.softplus(self.network(inputs)) + self.alpha_min
        return alpha.clamp_max(self.alpha_max)

    def select_active_preference(self, preference: torch.Tensor) -> torch.Tensor:
        """Return the two active SAR preference components."""
        return preference.index_select(-1, self._active_preference_indices)

    def expand_active_preference(
        self, active_preference: torch.Tensor
    ) -> torch.Tensor:
        """Map a two-dimensional sample to the environment preference."""
        preference = active_preference.new_zeros(
            *active_preference.shape[:-1], self.preference_dim
        )
        return preference.index_copy(
            -1, self._active_preference_indices, active_preference
        )


class PreferenceConditionedActor(nn.Module):
    """Shared Gaussian actor with separate observation/preference encoders."""

    def __init__(
        self,
        observation_dim: int,
        preference_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes: Sequence[int],
        preference_embedding_dim: int,
        agent_embedding_dim: int,
        min_scale: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        device,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("actor_hidden_sizes must contain at least one layer")
        self.n_agents = n_agents
        self.min_scale = float(min_scale)
        first_hidden = int(hidden_sizes[0])
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, first_hidden), nn.Tanh()
        ).to(device)
        self.preference_encoder = nn.Sequential(
            nn.Linear(preference_dim, preference_embedding_dim), nn.Tanh()
        ).to(device)
        self.agent_id = nn.Embedding(n_agents, agent_embedding_dim).to(device)
        fusion_sizes = tuple(int(v) for v in hidden_sizes[1:])
        self.policy_head = _mlp(
            first_hidden + preference_embedding_dim + agent_embedding_dim,
            fusion_sizes,
            2 * action_dim,
            device=device,
        )
        self.register_buffer(
            "action_low", action_low.detach().clone().to(device, torch.float32)
        )
        self.register_buffer(
            "action_high", action_high.detach().clone().to(device, torch.float32)
        )

    def parameters_for_distribution(
        self, observation: torch.Tensor, local_preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_embedding = self.observation_encoder(observation.to(torch.float32))
        pref_embedding = self.preference_encoder(local_preference.to(torch.float32))
        ids = _agent_embedding(self.agent_id, observation, self.n_agents)
        loc, raw_scale = self.policy_head(
            torch.cat((obs_embedding, pref_embedding, ids), dim=-1)
        ).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        return loc, scale

    def distribution(
        self, observation: torch.Tensor, local_preference: torch.Tensor
    ) -> TanhNormal:
        loc, scale = self.parameters_for_distribution(observation, local_preference)
        return TanhNormal(
            loc,
            scale,
            low=self.action_low,
            high=self.action_high,
            event_dims=1,
        )


class PCMAPolicyCore(nn.Module):
    """Collection policy that samples preferences before sampling actions."""

    def __init__(
        self,
        planner: PreferencePlanner,
        actor: PreferenceConditionedActor,
        use_global_preference_as_local: bool,
    ):
        super().__init__()
        self.planner = planner
        self.actor = actor
        self.use_global_preference_as_local = bool(
            use_global_preference_as_local
        )

    def forward(self, observation: torch.Tensor, environment_preference: torch.Tensor):
        global_preference = environment_preference.to(torch.float32).clone()
        deterministic = exploration_type() == ExplorationType.DETERMINISTIC
        if self.use_global_preference_as_local:
            # Preference-conditioned PPO ablation: bypass the stochastic
            # planner and give every agent the user-requested preference.
            local_preference = global_preference.clone()
            alpha = self.planner.select_active_preference(local_preference)
            preference_log_prob = local_preference.new_zeros(
                *local_preference.shape[:-1], 1
            )
        else:
            alpha = self.planner(observation, global_preference)
            preference_distribution = Dirichlet(alpha)
            if deterministic:
                active_local_preference = alpha / alpha.sum(
                    dim=-1, keepdim=True
                )
            else:
                active_local_preference = preference_distribution.sample()
            local_preference = self.planner.expand_active_preference(
                active_local_preference
            )
            preference_log_prob = preference_distribution.log_prob(
                active_local_preference
            ).unsqueeze(-1)

        action_distribution = self.actor.distribution(observation, local_preference)
        if deterministic:
            action = action_distribution.deterministic_sample
        else:
            action = action_distribution.rsample()
        action_log_prob = action_distribution.log_prob(action).unsqueeze(-1)
        return (
            global_preference,
            local_preference,
            alpha,
            preference_log_prob,
            action,
            action_log_prob,
        )


class CentralVectorCritics(nn.Module):
    """Central team critic and agent-wise vector critics."""

    def __init__(
        self,
        central_observation_dim: int,
        preference_dim: int,
        n_agents: int,
        hidden_sizes: Sequence[int],
        device,
    ):
        super().__init__()
        critic_input_dim = central_observation_dim + preference_dim
        self.n_agents = n_agents
        self.preference_dim = preference_dim
        self.team = _mlp(
            critic_input_dim,
            hidden_sizes,
            preference_dim,
            device=device,
        )
        self.individual = _mlp(
            critic_input_dim,
            hidden_sizes,
            n_agents * preference_dim,
            device=device,
        )

    def forward(
        self, central_observation: torch.Tensor, global_preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            (
                central_observation.to(torch.float32),
                global_preference.to(torch.float32),
            ),
            dim=-1,
        )
        team_value = self.team(inputs)
        individual_value = self.individual(inputs).reshape(
            *inputs.shape[:-1], self.n_agents, self.preference_dim
        )
        return team_value, individual_value


class PCMALoss(nn.Module):
    """Clipped PPO losses for actions and preferences plus vector critics."""

    def __init__(
        self,
        policy_core: PCMAPolicyCore,
        critics: CentralVectorCritics,
        group: str,
        n_agents: int,
        preference_dim: int,
        clip_epsilon: float,
        entropy_coef: float,
        planner_entropy_coef: float,
        critic_coef: float,
        individual_advantage_coef: float,
        diversity_coef: float,
        normalize_advantage: bool,
        loss_critic_type: str,
        state_key,
        state_ndim: int,
    ):
        super().__init__()
        self.actor = policy_core.actor
        self.planner = policy_core.planner
        self.use_global_preference_as_local = (
            policy_core.use_global_preference_as_local
        )
        self.critics = critics
        self.group = group
        self.n_agents = n_agents
        self.preference_dim = preference_dim
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.planner_entropy_coef = float(planner_entropy_coef)
        self.critic_coef = float(critic_coef)
        self.individual_advantage_coef = float(individual_advantage_coef)
        self.diversity_coef = float(diversity_coef)
        self.normalize_advantage = bool(normalize_advantage)
        self.loss_critic_type = loss_critic_type
        self.state_key = state_key
        self.state_ndim = state_ndim

    def _central_observation(self, td: TensorDictBase) -> torch.Tensor:
        if self.state_key is not None:
            state = td.get(self.state_key)
            return state.flatten(start_dim=-self.state_ndim)
        observation = td.get((self.group, "observation"))
        return observation.flatten(start_dim=-2)

    def _global_preference(self, td: TensorDictBase) -> torch.Tensor:
        key = (self.group, "global_preference")
        if key in td.keys(True, True):
            preference = td.get(key)
        else:
            preference = td.get((self.group, "preference"))
        if preference.shape[-2] == self.n_agents:
            preference = preference.mean(dim=-2)
        return preference.to(torch.float32)

    def _normalise(self, advantage: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantage:
            return advantage
        return (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

    def _distance(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_critic_type == "l1":
            return F.l1_loss(prediction, target)
        if self.loss_critic_type == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if self.loss_critic_type == "l2":
            return F.mse_loss(prediction, target)
        raise ValueError("loss_critic_type must be one of 'l1', 'l2', or 'smooth_l1'")

    def _expected_diversity(self, alpha: torch.Tensor) -> torch.Tensor:
        """Exact E[D_p] for independent per-agent Dirichlet variables."""
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        mean = alpha / alpha_0
        second_moment = alpha * (alpha + 1.0) / (alpha_0 * (alpha_0 + 1.0))
        squared_norm = second_moment.sum(dim=-1)
        pair_distance = (
            squared_norm.unsqueeze(-1)
            + squared_norm.unsqueeze(-2)
            - 2.0 * torch.matmul(mean, mean.transpose(-1, -2))
        )
        diagonal_mask = 1.0 - torch.eye(
            self.n_agents, device=alpha.device, dtype=alpha.dtype
        )
        pair_distance = pair_distance * diagonal_mask
        return pair_distance.sum(dim=(-2, -1)).mean() / (
            2.0 * self.n_agents * self.n_agents
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        observation = tensordict.get((self.group, "observation"))
        global_preference_by_agent = tensordict.get((self.group, "global_preference"))
        global_preference = self._global_preference(tensordict)
        local_preference = tensordict.get((self.group, "local_preference")).detach()

        # The stored local preference is treated as an action of the planner.
        # Detaching it keeps the actor and planner optimisation graphs separate.
        action_distribution = self.actor.distribution(observation, local_preference)
        action = tensordict.get((self.group, "action"))
        new_action_log_prob = action_distribution.log_prob(action).unsqueeze(-1)
        old_action_log_prob = tensordict.get((self.group, "log_prob")).detach()
        action_ratio = (new_action_log_prob - old_action_log_prob).exp()

        raw_actor_advantage = tensordict.get((self.group, "actor_advantage")).detach()
        actor_advantage = self._normalise(raw_actor_advantage)
        unclipped_actor = action_ratio * actor_advantage
        clipped_actor = (
            action_ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * actor_advantage
        )
        actor_objective = torch.minimum(unclipped_actor, clipped_actor).mean()
        loc, scale = self.actor.parameters_for_distribution(
            observation, local_preference
        )
        action_entropy = Normal(loc, scale).entropy().sum(dim=-1).mean()
        loss_actor = -actor_objective - self.entropy_coef * action_entropy

        raw_team_advantage = tensordict.get((self.group, "team_advantage")).detach()
        if self.use_global_preference_as_local:
            loss_planner = loss_actor.new_zeros(())
            planner_entropy = loss_actor.new_zeros(())
            diversity = loss_actor.new_zeros(())
            approx_planner_kl = loss_actor.new_zeros(())
        else:
            alpha = self.planner(observation, global_preference_by_agent)
            planner_distribution = Dirichlet(alpha)
            active_local_preference = self.planner.select_active_preference(
                local_preference
            )
            new_preference_log_prob = planner_distribution.log_prob(
                active_local_preference
            ).unsqueeze(-1)
            old_preference_log_prob = tensordict.get(
                (self.group, "preference_log_prob")
            ).detach()
            planner_ratio = (
                new_preference_log_prob - old_preference_log_prob
            ).exp()
            team_advantage = self._normalise(raw_team_advantage)
            unclipped_planner = planner_ratio * team_advantage
            clipped_planner = (
                planner_ratio.clamp(
                    1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                )
                * team_advantage
            )
            planner_objective = torch.minimum(
                unclipped_planner, clipped_planner
            ).mean()
            planner_entropy = planner_distribution.entropy().mean()
            diversity = self._expected_diversity(alpha)
            loss_planner = (
                -planner_objective
                - self.planner_entropy_coef * planner_entropy
                - self.diversity_coef * diversity
            )
            approx_planner_kl = (
                old_preference_log_prob - new_preference_log_prob
            ).mean().detach()

        central_observation = self._central_observation(tensordict)
        team_value, individual_value = self.critics(
            central_observation, global_preference
        )
        team_target = tensordict.get("pcma_team_value_target").detach()
        individual_target = tensordict.get("pcma_individual_value_target").detach()
        # Eq. (1) sums the individual vector-critic losses over agents.
        # _distance averages every tensor dimension, so multiplying by the
        # number of agents restores that paper-defined sum while retaining
        # the usual mean reduction over samples and objectives.
        loss_critic = self.critic_coef * (
            self._distance(team_value, team_target)
            + self.n_agents * self._distance(individual_value, individual_target)
        )

        approx_action_kl = (old_action_log_prob - new_action_log_prob).mean().detach()
        return TensorDict(
            {
                "loss_actor": loss_actor,
                "loss_planner": loss_planner,
                "loss_critic": loss_critic,
                "action_entropy": action_entropy.detach(),
                "planner_entropy": planner_entropy.detach(),
                "preference_diversity": diversity.detach(),
                "action_kl": approx_action_kl,
                "planner_kl": approx_planner_kl,
                "mean_team_advantage": raw_team_advantage.mean().detach(),
                "std_team_advantage": raw_team_advantage.std(unbiased=False).detach(),
                "mean_actor_advantage": raw_actor_advantage.mean().detach(),
                "std_actor_advantage": raw_actor_advantage.std(unbiased=False).detach(),
            },
            batch_size=[],
            device=loss_actor.device,
        )


class PCMA(Algorithm):
    """Continuous-action PCMA integrated through BenchMARL's on-policy loop."""

    def __init__(
        self,
        actor_hidden_sizes: Sequence[int],
        planner_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        preference_embedding_dim: int,
        agent_embedding_dim: int,
        clip_epsilon: float,
        entropy_coef: float,
        planner_entropy_coef: float,
        critic_coef: float,
        individual_advantage_coef: float,
        diversity_coef: float,
        lmbda: float,
        loss_critic_type: str,
        normalize_advantage: bool,
        alpha_min: float,
        alpha_max: float,
        min_action_scale: float,
        use_global_preference_as_local: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.actor_hidden_sizes = tuple(actor_hidden_sizes)
        self.planner_hidden_sizes = tuple(planner_hidden_sizes)
        self.critic_hidden_sizes = tuple(critic_hidden_sizes)
        self.preference_embedding_dim = int(preference_embedding_dim)
        self.agent_embedding_dim = int(agent_embedding_dim)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.planner_entropy_coef = float(planner_entropy_coef)
        self.critic_coef = float(critic_coef)
        self.individual_advantage_coef = float(individual_advantage_coef)
        self.diversity_coef = float(diversity_coef)
        self.lmbda = float(lmbda)
        self.loss_critic_type = loss_critic_type
        self.normalize_advantage = bool(normalize_advantage)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.min_action_scale = float(min_action_scale)
        self.use_global_preference_as_local = bool(
            use_global_preference_as_local
        )
        self._policy_cores: Dict[str, PCMAPolicyCore] = {}
        self._critics: Dict[str, CentralVectorCritics] = {}
        self._dimension_cache = {}

    def _dimensions(self, group: str):
        if group in self._dimension_cache:
            return self._dimension_cache[group]
        for key in ((group, "observation"), (group, "preference")):
            if key not in self.observation_spec.keys(True, True):
                raise KeyError(
                    f"PCMA requires observation key {key}; enable "
                    "condition_on_preference in the task configuration."
                )
        n_agents = len(self.group_map[group])
        observation_shape = self.observation_spec[group, "observation"].shape
        preference_shape = self.observation_spec[group, "preference"].shape
        action_shape = self.action_spec[group, "action"].shape
        if any(
            len(shape) != 2
            for shape in (observation_shape, preference_shape, action_shape)
        ):
            raise ValueError(
                "PCMA expects specs shaped (n_agents, feature_dim) for "
                "observations, preferences, and actions."
            )
        observation_dim = int(observation_shape[-1])
        preference_dim = int(preference_shape[-1])
        action_dim = int(action_shape[-1])
        if self.state_spec is None:
            central_observation_dim = n_agents * observation_dim
            state_key = None
            state_ndim = 0
        else:
            state_key = list(self.state_spec.keys(True, True))[0]
            state_shape = self.state_spec[state_key].shape
            central_observation_dim = int(torch.tensor(state_shape).prod().item())
            state_ndim = len(state_shape)
        result = (
            n_agents,
            observation_dim,
            preference_dim,
            action_dim,
            central_observation_dim,
            state_key,
            state_ndim,
        )
        self._dimension_cache[group] = result
        return result

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("This PCMA implementation is continuous-only")
        (
            n_agents,
            observation_dim,
            preference_dim,
            action_dim,
            _,
            _,
            _,
        ) = self._dimensions(group)
        action_spec = self.action_spec[group, "action"]
        planner = PreferencePlanner(
            observation_dim=observation_dim,
            preference_dim=preference_dim,
            n_agents=n_agents,
            hidden_sizes=self.planner_hidden_sizes,
            agent_embedding_dim=self.agent_embedding_dim,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            device=self.device,
        )
        actor = PreferenceConditionedActor(
            observation_dim=observation_dim,
            preference_dim=preference_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_sizes=self.actor_hidden_sizes,
            preference_embedding_dim=self.preference_embedding_dim,
            agent_embedding_dim=self.agent_embedding_dim,
            min_scale=self.min_action_scale,
            action_low=action_spec.space.low,
            action_high=action_spec.space.high,
            device=self.device,
        )
        core = PCMAPolicyCore(
            planner=planner,
            actor=actor,
            use_global_preference_as_local=self.use_global_preference_as_local,
        ).to(self.device)
        self._policy_cores[group] = core
        return TensorDictModule(
            core,
            in_keys=[
                (group, "observation"),
                (group, "preference"),
            ],
            out_keys=[
                (group, "global_preference"),
                (group, "local_preference"),
                (group, "preference_alpha"),
                (group, "preference_log_prob"),
                (group, "action"),
                (group, "log_prob"),
            ],
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return policy_for_loss

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[PCMALoss, bool]:
        if not continuous:
            raise NotImplementedError("This PCMA implementation is continuous-only")
        (
            n_agents,
            _,
            preference_dim,
            _,
            central_observation_dim,
            state_key,
            state_ndim,
        ) = self._dimensions(group)
        critics = CentralVectorCritics(
            central_observation_dim=central_observation_dim,
            preference_dim=preference_dim,
            n_agents=n_agents,
            hidden_sizes=self.critic_hidden_sizes,
            device=self.device,
        ).to(self.device)
        self._critics[group] = critics
        loss = PCMALoss(
            policy_core=self._policy_cores[group],
            critics=critics,
            group=group,
            n_agents=n_agents,
            preference_dim=preference_dim,
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            planner_entropy_coef=self.planner_entropy_coef,
            critic_coef=self.critic_coef,
            individual_advantage_coef=self.individual_advantage_coef,
            diversity_coef=self.diversity_coef,
            normalize_advantage=self.normalize_advantage,
            loss_critic_type=self.loss_critic_type,
            state_key=state_key,
            state_ndim=state_ndim,
        ).to(self.device)
        return loss, False

    def _get_parameters(self, group: str, loss: PCMALoss) -> Dict[str, Iterable]:
        parameters = {
            "loss_actor": loss.actor.parameters(),
            "loss_critic": loss.critics.parameters(),
        }
        if not self.use_global_preference_as_local:
            parameters["loss_planner"] = loss.planner.parameters()
        return parameters

    def _central_observation(self, group: str, td: TensorDictBase) -> torch.Tensor:
        *_, state_key, state_ndim = self._dimensions(group)
        if state_key is not None:
            return td.get(state_key).flatten(start_dim=-state_ndim)
        return td.get((group, "observation")).flatten(start_dim=-2)

    def _global_preference(self, group: str, td: TensorDictBase) -> torch.Tensor:
        key = (group, "global_preference")
        if key in td.keys(True, True):
            preference = td.get(key)
        else:
            preference = td.get((group, "preference"))
        n_agents = len(self.group_map[group])
        if preference.shape[-2] == n_agents:
            preference = preference.mean(dim=-2)
        return preference.to(torch.float32)

    def _vector_reward(self, group: str, next_td: TensorDictBase) -> torch.Tensor:
        candidates = (
            (group, "reward_vector"),
            (group, "vector_reward"),
            "reward_vector",
            "vector_reward",
            "mo_reward",
        )
        preference_dim = self._dimensions(group)[2]
        for key in candidates:
            if key not in next_td.keys(True, True):
                continue
            reward = next_td.get(key).to(torch.float32)
            if reward.shape[-1] == preference_dim:
                return reward
        raise KeyError(
            "PCMA requires an unscalarised vector reward under reward_vector, "
            "vector_reward, or mo_reward in the next TensorDict."
        )

    def _episode_flag(
        self, group: str, next_td: TensorDictBase, name: str
    ) -> torch.Tensor | None:
        if name in next_td.keys(True, True):
            return next_td.get(name).to(torch.bool)
        group_key = (group, name)
        if group_key in next_td.keys(True, True):
            return next_td.get(group_key).to(torch.bool).any(dim=-2)
        return None

    def _termination_masks(
        self, group: str, next_td: TensorDictBase
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the bootstrap and trace masks used by paper-style GAE.

        ``terminated`` controls whether the one-step value target may bootstrap.
        ``done`` additionally includes time-limit truncations and controls whether
        the recursive GAE trace may cross an episode boundary.
        """

        terminated = self._episode_flag(group, next_td, "terminated")
        done = self._episode_flag(group, next_td, "done")
        truncated = self._episode_flag(group, next_td, "truncated")

        if terminated is None and done is None:
            raise KeyError("PCMA could not find terminated/done in the batch")
        if terminated is None:
            # Environments exposing only done cannot distinguish true terminal
            # states from truncations, so use the conservative no-bootstrap mask.
            terminated = done
        if done is None:
            done = terminated
            if truncated is not None:
                done = torch.logical_or(done, truncated)

        return terminated.to(torch.float32), done.to(torch.float32)

    @staticmethod
    def _time_dim(batch: TensorDictBase) -> int:
        """Resolve TorchRL's rollout time dimension.

        SyncDataCollector returns vectorized rollouts as
        ``[*env_batch, time]`` and names the last batch dimension ``"time"``.
        Falling back to the last batch dimension also supports unnamed
        TensorDicts and the single-environment case.
        """

        names = getattr(batch, "names", None)
        if names is not None and "time" in names:
            return names.index("time")
        return len(batch.batch_size) - 1

    def _gae(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        terminated: torch.Tensor,
        done: torch.Tensor,
        time_dim: int,
    ) -> torch.Tensor:
        """Compute vector GAE without mixing independent environments.

        True terminations suppress value bootstrapping. Both terminations and
        time-limit truncations stop the recursive eligibility trace, matching
        PPO/GAE semantics while retaining a valid time-limit bootstrap.
        """

        if reward.shape != value.shape or reward.shape != next_value.shape:
            raise ValueError(
                "PCMA GAE expects reward, value, and next_value to have "
                f"identical shapes, got {reward.shape}, {value.shape}, "
                f"and {next_value.shape}."
            )
        if time_dim < 0:
            time_dim += reward.ndim
        if not 0 <= time_dim < reward.ndim:
            raise IndexError(
                f"Invalid PCMA GAE time dimension {time_dim} for {reward.shape}"
            )

        while terminated.ndim < reward.ndim:
            terminated = terminated.unsqueeze(-1)
        while done.ndim < reward.ndim:
            done = done.unsqueeze(-1)

        delta = (
            reward
            + self.experiment_config.gamma * (1.0 - terminated) * next_value
            - value
        )
        delta_by_time = delta.movedim(time_dim, 0)
        done_by_time = done.movedim(time_dim, 0)
        advantage_by_time = torch.zeros_like(delta_by_time)
        running = torch.zeros_like(delta_by_time[0])
        coefficient = self.experiment_config.gamma * self.lmbda
        for time_index in range(delta_by_time.shape[0] - 1, -1, -1):
            running = (
                delta_by_time[time_index]
                + coefficient * (1.0 - done_by_time[time_index]) * running
            )
            advantage_by_time[time_index] = running
        return advantage_by_time.movedim(0, time_dim)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        if len(batch.batch_size) == 0:
            raise ValueError("PCMA expects an on-policy rollout with a time dimension")
        next_td = batch.get("next")
        critics = self._critics[group]
        n_agents = self._dimensions(group)[0]
        time_dim = self._time_dim(batch)
        with torch.no_grad():
            global_preference = self._global_preference(group, batch)
            next_global_preference = self._global_preference(group, next_td)
            current_team_value, current_individual_value = critics(
                self._central_observation(group, batch), global_preference
            )
            next_team_value, next_individual_value = critics(
                self._central_observation(group, next_td),
                next_global_preference,
            )
            individual_reward = self._vector_reward(group, next_td)
            if individual_reward.shape[-2] != n_agents:
                individual_reward = individual_reward.unsqueeze(-2).expand(
                    *individual_reward.shape[:-1], n_agents, -1
                )
            team_reward = individual_reward.mean(dim=-2)
            terminated, done = self._termination_masks(group, next_td)
            team_vector_advantage = self._gae(
                team_reward,
                current_team_value,
                next_team_value,
                terminated,
                done,
                time_dim,
            )
            individual_vector_advantage = self._gae(
                individual_reward,
                current_individual_value,
                next_individual_value,
                terminated,
                done,
                time_dim,
            )
            team_advantage = (team_vector_advantage * global_preference).sum(
                dim=-1, keepdim=True
            )
            local_preference = batch.get((group, "local_preference"))
            individual_utility_advantage = (
                individual_vector_advantage * local_preference
            ).sum(dim=-1, keepdim=True)
            actor_advantage = (
                team_advantage.unsqueeze(-2).expand(
                    *team_advantage.shape[:-1], n_agents, 1
                )
                + self.individual_advantage_coef * individual_utility_advantage
            )
            team_advantage_by_agent = team_advantage.unsqueeze(-2).expand(
                *team_advantage.shape[:-1], n_agents, 1
            )

            batch.set((group, "team_advantage"), team_advantage_by_agent)
            batch.set((group, "actor_advantage"), actor_advantage)
            batch.set(
                "pcma_team_value_target",
                team_vector_advantage + current_team_value,
            )
            batch.set(
                "pcma_individual_value_target",
                individual_vector_advantage + current_individual_value,
            )
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        if self.use_global_preference_as_local:
            return loss_vals.select("loss_actor", "loss_critic")
        return loss_vals.select("loss_actor", "loss_planner", "loss_critic")


@dataclass
class PCMAConfig(AlgorithmConfig):
    """Configuration with paper-like defaults and no external YAML file."""

    actor_hidden_sizes: Tuple[int, ...] = (64, 64)
    planner_hidden_sizes: Tuple[int, ...] = (64, 64)
    critic_hidden_sizes: Tuple[int, ...] = (128, 128)
    preference_embedding_dim: int = 32
    agent_embedding_dim: int = 8
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    planner_entropy_coef: float = 0.0
    critic_coef: float = 0.5
    individual_advantage_coef: float = 0.2
    diversity_coef: float = 0.02
    lmbda: float = 0.95
    loss_critic_type: str = "l2"
    normalize_advantage: bool = True
    alpha_min: float = 0.05
    alpha_max: float = 100.0
    min_action_scale: float = 1e-4
    use_global_preference_as_local: bool = False

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return PCMA

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the standalone continuous-action PCMA implementation"
    )
    parser.add_argument("--env-path", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("moroom128cpu", "moempty128cpu", "momaze128cpu"),
        default="moroom128cpu",
    )
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diversity-coef", type=float, default=0.02)
    parser.add_argument("--individual-advantage-coef", type=float, default=0.2)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    # Imports are local so importing this algorithm does not initialize Unity
    # or pull in the command-line runner's optional dependencies.
    from benchmarl.environments import SARTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    args = _parse_args()
    env_path = args.env_path.expanduser().resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Unity environment does not exist: {env_path}")

    task = {
        "moroom128cpu": SARTask.MOROOM128CPU,
        "moempty128cpu": SARTask.MOEMPTY128CPU,
        "momaze128cpu": SARTask.MOMAZE128CPU,
    }[args.task].get_from_yaml()
    task.config["env_path"] = str(env_path)
    task.config["condition_on_preference"] = True
    task.config["fixed_preference"] = None
    task.config["u_sample_start"] = 0.0

    algorithm_config = PCMAConfig(
        diversity_coef=args.diversity_coef,
        individual_advantage_coef=args.individual_advantage_coef,
    )
    experiment_config = ExperimentConfig.get_from_yaml()
    device = _resolve_device(args.device)
    experiment_config.train_device = device
    experiment_config.buffer_device = device
    experiment_config.sampling_device = "cpu"
    experiment_config.max_n_frames = 2_000 if args.smoke_test else args.frames
    experiment_config.max_n_iters = None
    experiment_config.gamma = 0.99
    experiment_config.lr = 3e-4
    experiment_config.on_policy_n_envs_per_worker = 1
    experiment_config.on_policy_collected_frames_per_batch = (
        256 if args.smoke_test else 4_000
    )
    experiment_config.on_policy_minibatch_size = 64 if args.smoke_test else 500
    experiment_config.on_policy_n_minibatch_iters = 2 if args.smoke_test else 4
    experiment_config.parallel_collection = False
    experiment_config.parallel_evaluation = False
    experiment_config.evaluation = not args.smoke_test
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_at_end = True

    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()
