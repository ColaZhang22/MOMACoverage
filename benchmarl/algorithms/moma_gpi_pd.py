"""Continuous-action multi-agent GPI-PD for BenchMARL.

The original GPI-PD algorithm uses discrete Q-learning.  This adaptation uses
a preference-conditioned deterministic actor and centralised vector critics:

* each support weight defines a candidate continuous joint policy;
* GPI evaluates those candidates for the requested preference and executes
  the joint action with the largest scalarised vector Q-value;
* the critics learn from the ordinary TD3 vector target;
* replay priority is computed from the GPI envelope TD error, as in GPI-PD.

The Dyna component from the original implementation is intentionally omitted:
injecting synthetic transitions requires changing BenchMARL's collector and
replay-buffer loop, while this module is designed to be added without touching
existing framework files.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.modules import AdditiveGaussianModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.moma_ac import (
    CentralisedVectorCritic,
    MultiHeadedActor,
)
from benchmarl.models.common import ModelConfig


def _normalise_support(
    support: Sequence[Sequence[float]] | torch.Tensor,
    n_objectives: int,
    *,
    device,
) -> torch.Tensor:
    weights = torch.as_tensor(support, dtype=torch.float32, device=device)
    if weights.numel() == 0:
        return torch.empty((0, n_objectives), device=device)
    if weights.ndim != 2 or weights.shape[-1] != n_objectives:
        raise ValueError(
            "weight_support must have shape "
            f"(n_weights, {n_objectives}), got {tuple(weights.shape)}"
        )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weight_support must contain finite non-negative values")
    totals = weights.sum(dim=-1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("every support weight must have a positive sum")
    weights = weights / totals
    unique: List[torch.Tensor] = []
    for weight in weights:
        if not any(torch.allclose(weight, old, atol=1e-7) for old in unique):
            unique.append(weight)
    return torch.stack(unique)


class ContinuousMultiAgentGPIPolicy(nn.Module):
    """GPI selector over preference-conditioned continuous joint policies."""

    def __init__(
        self,
        actor: MultiHeadedActor,
        critic_1: CentralisedVectorCritic,
        critic_2: CentralisedVectorCritic,
        weight_support: Sequence[Sequence[float]],
        n_agents: int,
        n_objectives: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        include_current_preference: bool,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.n_agents = int(n_agents)
        self.n_objectives = int(n_objectives)
        self.include_current_preference = bool(include_current_preference)
        device = next(actor.parameters()).device
        self.register_buffer(
            "action_low",
            action_low.detach().clone().to(device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            action_high.detach().clone().to(device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_support",
            _normalise_support(
                weight_support, n_objectives, device=device
            ),
        )

    def set_weight_support(
        self, support: Sequence[Sequence[float]] | torch.Tensor
    ) -> None:
        self.weight_support = _normalise_support(
            support,
            self.n_objectives,
            device=self.weight_support.device,
        )

    def bounded_action(
        self,
        actor: MultiHeadedActor,
        observation: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        unit_action = actor(observation, preference).tanh()
        return self.action_low + (unit_action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

    def candidate_weights(self, preference: torch.Tensor) -> torch.Tensor:
        """Return ``(..., n_candidates, n_agents, n_objectives)``."""
        leading = preference.shape[:-2]
        if self.weight_support.shape[0] == 0:
            return preference.unsqueeze(-3)
        support = self.weight_support.view(
            *((1,) * len(leading)),
            self.weight_support.shape[0],
            1,
            self.n_objectives,
        ).expand(
            *leading,
            self.weight_support.shape[0],
            self.n_agents,
            self.n_objectives,
        )
        if self.include_current_preference:
            support = torch.cat((support, preference.unsqueeze(-3)), dim=-3)
        return support

    @staticmethod
    def _central_from_input(
        observation: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        if state is None:
            return observation
        return state.reshape(*observation.shape[:-2], -1).unsqueeze(-2)

    @staticmethod
    def _expand_candidates(
        value: torch.Tensor, n_candidates: int
    ) -> torch.Tensor:
        return value.unsqueeze(-3).expand(
            *value.shape[:-2],
            n_candidates,
            *value.shape[-2:],
        )

    @staticmethod
    def _select_clipped_vector(
        q_1: torch.Tensor,
        q_2: torch.Tensor,
        scalarisation_weight: torch.Tensor,
    ) -> torch.Tensor:
        utility_1 = (q_1 * scalarisation_weight).sum(
            dim=-1, keepdim=True
        )
        utility_2 = (q_2 * scalarisation_weight).sum(
            dim=-1, keepdim=True
        )
        return torch.where(utility_1 <= utility_2, q_1, q_2)

    def evaluate_candidates(
        self,
        actor: MultiHeadedActor,
        critic_1: CentralisedVectorCritic,
        critic_2: CentralisedVectorCritic,
        observation: torch.Tensor,
        central_observation: torch.Tensor,
        requested_preference: torch.Tensor,
        *,
        add_target_noise: bool = False,
        target_noise: float = 0.0,
        target_noise_clip: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate support policies and return actions, Q vectors and weights."""
        candidates = self.candidate_weights(requested_preference)
        n_candidates = candidates.shape[-3]
        expanded_observation = self._expand_candidates(
            observation, n_candidates
        )
        expanded_central = self._expand_candidates(
            central_observation, n_candidates
        )
        actions = self.bounded_action(
            actor, expanded_observation, candidates
        )
        if add_target_noise and target_noise > 0:
            noise = torch.randn_like(actions) * target_noise
            noise = noise.clamp(-target_noise_clip, target_noise_clip)
            actions = (actions + noise).clamp(
                self.action_low, self.action_high
            )
        group_candidates = candidates.mean(dim=-2)
        q_1 = critic_1(expanded_central, actions, group_candidates)
        q_2 = critic_2(expanded_central, actions, group_candidates)
        requested = requested_preference.mean(dim=-2).unsqueeze(-2)
        vectors = self._select_clipped_vector(
            q_1, q_2, requested
        )
        return actions, vectors, group_candidates

    @staticmethod
    def select_best_candidate(
        actions: torch.Tensor,
        vectors: torch.Tensor,
        requested_preference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requested = requested_preference.mean(dim=-2)
        utilities = (vectors * requested.unsqueeze(-2)).sum(dim=-1)
        policy_index = utilities.argmax(dim=-1)
        chosen_action = actions.gather(
            -3,
            policy_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                *policy_index.shape,
                1,
                actions.shape[-2],
                actions.shape[-1],
            ),
        ).squeeze(-3)
        chosen_vector = vectors.gather(
            -2,
            policy_index.unsqueeze(-1).unsqueeze(-1).expand(
                *policy_index.shape, 1, vectors.shape[-1]
            ),
        ).squeeze(-2)
        return chosen_action, chosen_vector, policy_index

    def forward(
        self,
        observation: torch.Tensor,
        preference: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        central = self._central_from_input(observation, state)
        actions, vectors, _ = self.evaluate_candidates(
            self.actor,
            self.critic_1,
            self.critic_2,
            observation,
            central,
            preference,
        )
        return self.select_best_candidate(actions, vectors, preference)


class MomaGPIPDLoss(nn.Module):
    """TD3-style vector loss with GPI-PD envelope priorities."""

    def __init__(
        self,
        policy: ContinuousMultiAgentGPIPolicy,
        group: str,
        n_agents: int,
        n_objectives: int,
        gamma: float,
        polyak_tau: float,
        policy_delay: int,
        target_policy_noise: float,
        target_noise_clip: float,
        loss_function: str,
        min_priority: float,
        priority_exponent: float,
        support_sample_probability: float,
        state_key=None,
    ) -> None:
        super().__init__()
        if min_priority <= 0:
            raise ValueError("min_priority must be positive")
        if priority_exponent <= 0:
            raise ValueError("priority_exponent must be positive")
        if not 0.0 <= support_sample_probability <= 1.0:
            raise ValueError("support_sample_probability must be in [0, 1]")
        self.policy = policy
        self.target_actor = copy.deepcopy(policy.actor).requires_grad_(False)
        self.target_critic_1 = copy.deepcopy(
            policy.critic_1
        ).requires_grad_(False)
        self.target_critic_2 = copy.deepcopy(
            policy.critic_2
        ).requires_grad_(False)
        self.group = group
        self.n_agents = int(n_agents)
        self.n_objectives = int(n_objectives)
        self.gamma = float(gamma)
        self.polyak_tau = float(polyak_tau)
        self.policy_delay = max(int(policy_delay), 1)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.loss_function = loss_function
        self.min_priority = float(min_priority)
        self.priority_exponent = float(priority_exponent)
        self.support_sample_probability = float(
            support_sample_probability
        )
        self.state_key = state_key
        device = next(policy.actor.parameters()).device
        self.register_buffer(
            "_updates", torch.zeros((), dtype=torch.long, device=device)
        )
        self.last_actor_update = False
        self._pending_target_update = False

    @property
    def online_critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.policy.critic_1.parameters()
        yield from self.policy.critic_2.parameters()

    def set_weight_support(
        self, support: Sequence[Sequence[float]] | torch.Tensor
    ) -> None:
        self.policy.set_weight_support(support)

    def _central_observation(self, td: TensorDictBase) -> torch.Tensor:
        if self.state_key is None:
            return td.get((self.group, "observation"))
        state = td.get(self.state_key)
        return state.reshape(*td.batch_size, -1).unsqueeze(-2)

    def _training_preference(self, td: TensorDictBase) -> torch.Tensor:
        preference = td.get((self.group, "preference")).to(torch.float32)
        if preference.shape[-2:] != (self.n_agents, self.n_objectives):
            raise ValueError(
                "preference must end in "
                f"({self.n_agents}, {self.n_objectives}), got {preference.shape}"
            )
        support = self.policy.weight_support
        if (
            support.shape[0] == 0
            or self.support_sample_probability == 0
        ):
            return preference
        batch_shape = preference.shape[:-2]
        index = torch.randint(
            support.shape[0], batch_shape, device=preference.device
        )
        sampled = support[index].unsqueeze(-2).expand(
            *batch_shape, self.n_agents, self.n_objectives
        )
        replace = (
            torch.rand(batch_shape, device=preference.device)
            < self.support_sample_probability
        )
        return torch.where(
            replace.unsqueeze(-1).unsqueeze(-1),
            sampled,
            preference,
        )

    def _vector_reward(self, next_td: TensorDictBase) -> torch.Tensor:
        for key in (
            (self.group, "reward_vector"),
            (self.group, "vector_reward"),
            (self.group, "mo_reward"),
            "reward_vector",
            "vector_reward",
            "mo_reward",
        ):
            if key not in next_td.keys(True, True):
                continue
            reward = next_td.get(key).to(torch.float32)
            if reward.shape[-1] != self.n_objectives:
                continue
            if reward.ndim >= 2 and reward.shape[-2] == self.n_agents:
                reward = reward.mean(dim=-2)
            return reward
        raise KeyError(
            "MomaGPI-PD requires vector reward under reward_vector, "
            "vector_reward, or mo_reward"
        )

    def _done(self, next_td: TensorDictBase) -> torch.Tensor:
        for key in ("terminated", "done"):
            if key in next_td.keys(True, True):
                return next_td.get(key).to(torch.float32)
        for key in (
            (self.group, "terminated"),
            (self.group, "done"),
        ):
            if key in next_td.keys(True, True):
                return next_td.get(key).any(dim=-2).to(torch.float32)
        raise KeyError("MomaGPI-PD could not find done or terminated")

    def _distance(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_function == "l1":
            return F.l1_loss(prediction, target)
        if self.loss_function == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if self.loss_function == "l2":
            return F.mse_loss(prediction, target)
        raise ValueError(f"unknown loss_function: {self.loss_function}")

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for target_module, online_module in (
            (self.target_actor, self.policy.actor),
            (self.target_critic_1, self.policy.critic_1),
            (self.target_critic_2, self.policy.critic_2),
        ):
            for target, online in zip(
                target_module.parameters(), online_module.parameters()
            ):
                target.lerp_(online, self.polyak_tau)

    def _standard_bootstrap(
        self,
        next_observation: torch.Tensor,
        next_central: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        action = self.policy.bounded_action(
            self.target_actor, next_observation, preference
        )
        noise = torch.randn_like(action) * self.target_policy_noise
        noise = noise.clamp(
            -self.target_noise_clip, self.target_noise_clip
        )
        action = (action + noise).clamp(
            self.policy.action_low, self.policy.action_high
        )
        group_preference = preference.mean(dim=-2)
        q_1 = self.target_critic_1(
            next_central, action, group_preference
        )
        q_2 = self.target_critic_2(
            next_central, action, group_preference
        )
        return self.policy._select_clipped_vector(
            q_1, q_2, group_preference
        )

    def _envelope_bootstrap(
        self,
        next_observation: torch.Tensor,
        next_central: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        actions, vectors, _ = self.policy.evaluate_candidates(
            self.target_actor,
            self.target_critic_1,
            self.target_critic_2,
            next_observation,
            next_central,
            preference,
            add_target_noise=True,
            target_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
        )
        _, vector, _ = self.policy.select_best_candidate(
            actions, vectors, preference
        )
        return vector

    def _critic_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        central = self._central_observation(td)
        action = td.get((self.group, "action"))
        group_preference = preference.mean(dim=-2)
        q_1 = self.policy.critic_1(
            central, action, group_preference
        )
        q_2 = self.policy.critic_2(
            central, action, group_preference
        )
        with torch.no_grad():
            next_td = td.get("next")
            next_observation = next_td.get(
                (self.group, "observation")
            )
            next_central = self._central_observation(next_td)
            reward = self._vector_reward(next_td)
            done = self._done(next_td)
            standard = self._standard_bootstrap(
                next_observation, next_central, preference
            )
            envelope = self._envelope_bootstrap(
                next_observation, next_central, preference
            )
            standard_target = (
                reward + self.gamma * (1.0 - done) * standard
            )
            priority_target = (
                reward + self.gamma * (1.0 - done) * envelope
            )
        loss = self._distance(q_1, standard_target) + self._distance(
            q_2, standard_target
        )
        vector_error = torch.stack(
            (
                (priority_target - q_1.detach()).abs(),
                (priority_target - q_2.detach()).abs(),
            )
        ).amax(dim=0)
        scalar_error = (
            vector_error * group_preference
        ).sum(dim=-1, keepdim=True).abs()
        utility = (q_1.detach() * group_preference).sum(dim=-1).mean()
        return loss, scalar_error, utility

    def _actor_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> torch.Tensor:
        observation = td.get((self.group, "observation"))
        central = self._central_observation(td)
        action = self.policy.bounded_action(
            self.policy.actor, observation, preference
        )
        critic_parameters = list(self.online_critic_parameters)
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            q_vector = self.policy.critic_1(
                central, action, preference.mean(dim=-2)
            )
            return -(
                q_vector * preference.mean(dim=-2)
            ).sum(dim=-1).mean()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

    def forward(self, td: TensorDictBase) -> TensorDict:
        if self._pending_target_update:
            self._soft_update_targets()
            self._pending_target_update = False
        self._updates += 1
        self.last_actor_update = (
            int(self._updates.item()) % self.policy_delay == 0
        )
        preference = self._training_preference(td)
        critic_loss, scalar_error, utility = self._critic_loss(
            td, preference
        )
        if self.last_actor_update:
            actor_loss = self._actor_loss(td, preference)
            self._pending_target_update = True
        else:
            actor_loss = critic_loss.new_zeros(())
        priority = scalar_error.clamp_min(self.min_priority).pow(
            self.priority_exponent
        )
        td.set(
            (self.group, "td_error"),
            priority.unsqueeze(-2).expand(
                *priority.shape[:-1], self.n_agents, 1
            ),
        )
        return TensorDict(
            {
                "loss_actor": actor_loss,
                "loss_critic": critic_loss,
                "td_error": scalar_error.mean(),
                "priority": priority.mean(),
                "utility": utility,
                "grad_norm_loss_actor": actor_loss.new_zeros(()),
            },
            batch_size=[],
            device=critic_loss.device,
        )


class MomaGPIPD(Algorithm):
    """Continuous-action multi-agent GPI-PD."""

    def __init__(
        self,
        actor_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        weight_support: Sequence[Sequence[float]],
        include_current_preference: bool,
        support_sample_probability: float,
        loss_function: str,
        policy_delay: int,
        target_policy_noise: float,
        target_noise_clip: float,
        exploration_noise: float,
        exploration_noise_end: float,
        min_priority: float,
        priority_exponent: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.actor_hidden_sizes = tuple(actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(critic_hidden_sizes)
        self.weight_support = [list(weight) for weight in weight_support]
        self.include_current_preference = include_current_preference
        self.support_sample_probability = support_sample_probability
        self.loss_function = loss_function
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.exploration_noise_end = exploration_noise_end
        self.min_priority = min_priority
        self.priority_exponent = priority_exponent
        self._gpi_policies: Dict[str, ContinuousMultiAgentGPIPolicy] = {}

    def _dimensions(self, group: str):
        for key in ((group, "observation"), (group, "preference")):
            if key not in self.observation_spec.keys(True, True):
                raise KeyError(
                    f"MomaGPI-PD requires {key}; enable preference conditioning"
                )
        n_agents = len(self.group_map[group])
        observation_shape = self.observation_spec[
            group, "observation"
        ].shape
        preference_shape = self.observation_spec[
            group, "preference"
        ].shape
        action_shape = self.action_spec[group, "action"].shape
        if any(
            len(shape) != 2
            for shape in (
                observation_shape,
                preference_shape,
                action_shape,
            )
        ):
            raise ValueError(
                "MomaGPI-PD expects observation, preference and continuous "
                "action specs shaped (n_agents, feature_dim)"
            )
        if self.state_spec is None:
            central_dim = n_agents * int(observation_shape[-1])
            state_key = None
        else:
            state_key = list(self.state_spec.keys(True, True))[0]
            central_dim = int(
                torch.tensor(self.state_spec[state_key].shape).prod().item()
            )
        return (
            n_agents,
            int(observation_shape[-1]),
            int(preference_shape[-1]),
            int(action_shape[-1]),
            central_dim,
            state_key,
        )

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError(
                "MomaGPI-PD supports continuous actions only"
            )
        (
            n_agents,
            observation_dim,
            n_objectives,
            action_dim,
            central_dim,
            state_key,
        ) = self._dimensions(group)
        actor = MultiHeadedActor(
            observation_dim=observation_dim,
            preference_dim=n_objectives,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_sizes=self.actor_hidden_sizes,
            device=self.device,
        )
        critic_args = dict(
            central_observation_dim=central_dim,
            joint_action_dim=n_agents * action_dim,
            preference_dim=n_objectives,
            hidden_sizes=self.critic_hidden_sizes,
            device=self.device,
        )
        action_spec = self.action_spec[group, "action"]
        policy = ContinuousMultiAgentGPIPolicy(
            actor=actor,
            critic_1=CentralisedVectorCritic(**critic_args),
            critic_2=CentralisedVectorCritic(**critic_args),
            weight_support=self.weight_support,
            n_agents=n_agents,
            n_objectives=n_objectives,
            action_low=action_spec.space.low,
            action_high=action_spec.space.high,
            include_current_preference=self.include_current_preference,
        )
        self._gpi_policies[group] = policy
        in_keys = [(group, "observation"), (group, "preference")]
        if state_key is not None:
            in_keys.append(state_key)
        return TensorDictModule(
            policy,
            in_keys=in_keys,
            out_keys=[
                (group, "action"),
                (group, "chosen_action_value_vector"),
                (group, "gpi_policy_index"),
            ],
        )

    def _get_policy_for_collection(
        self,
        policy_for_loss: TensorDictModule,
        group: str,
        continuous: bool,
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError(
                "MomaGPI-PD supports continuous actions only"
            )
        noise = AdditiveGaussianModule(
            spec=self.action_spec,
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            sigma_init=self.exploration_noise,
            sigma_end=self.exploration_noise_end,
            device=self.device,
        )
        return TensorDictSequential(policy_for_loss, noise)

    def _get_loss(
        self,
        group: str,
        policy_for_loss: TensorDictModule,
        continuous: bool,
    ) -> Tuple[MomaGPIPDLoss, bool]:
        if not continuous:
            raise NotImplementedError(
                "MomaGPI-PD supports continuous actions only"
            )
        n_agents, _, n_objectives, _, _, state_key = self._dimensions(group)
        loss = MomaGPIPDLoss(
            policy=self._gpi_policies[group],
            group=group,
            n_agents=n_agents,
            n_objectives=n_objectives,
            gamma=self.experiment_config.gamma,
            polyak_tau=self.experiment_config.polyak_tau,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            loss_function=self.loss_function,
            min_priority=self.min_priority,
            priority_exponent=self.priority_exponent,
            support_sample_probability=self.support_sample_probability,
            state_key=state_key,
        ).to(self.device)
        return loss, False

    def _get_parameters(
        self, group: str, loss: MomaGPIPDLoss
    ) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "loss_actor": loss.policy.actor.parameters(),
            "loss_critic": loss.online_critic_parameters,
        }

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss = self.get_loss_and_updater(group)[0]
        optimisable = loss_vals.select("loss_actor", "loss_critic")
        if not loss.last_actor_update:
            optimisable = optimisable.exclude("loss_actor")
        return optimisable

    def process_batch(
        self, group: str, batch: TensorDictBase
    ) -> TensorDictBase:
        return batch

    def set_weight_support(
        self, support: Sequence[Sequence[float]] | torch.Tensor
    ) -> None:
        """Update the GPI support after an outer-loop support-set iteration."""
        if isinstance(support, torch.Tensor):
            raw = support.detach().cpu().tolist()
        else:
            raw = [list(weight) for weight in support]
        self.weight_support = raw
        for policy in self._gpi_policies.values():
            policy.set_weight_support(raw)
        for loss, _ in self._losses_and_updaters.values():
            loss.set_weight_support(raw)


@dataclass
class MomaGPIPDConfig(AlgorithmConfig):
    """Configuration for continuous-action :class:`MomaGPIPD`."""

    actor_hidden_sizes: List[int] = MISSING
    critic_hidden_sizes: List[int] = MISSING
    weight_support: List[List[float]] = MISSING
    include_current_preference: bool = MISSING
    support_sample_probability: float = MISSING
    loss_function: str = MISSING
    policy_delay: int = MISSING
    target_policy_noise: float = MISSING
    target_noise_clip: float = MISSING
    exploration_noise: float = MISSING
    exploration_noise_end: float = MISSING
    min_priority: float = MISSING
    priority_exponent: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MomaGPIPD

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True

