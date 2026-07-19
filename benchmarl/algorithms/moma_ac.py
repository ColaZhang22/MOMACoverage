"""Paper-faithful MOMA-AC for continuous multi-objective MARL.

This module implements the MOMA-TD3 and MOMA-DDPG instances described in:

    Callaghan, Mason, and Mannion, "MOMA-AC: A Preference-Driven
    Actor-Critic Framework for Continuous Multi-Objective Multi-Agent
    Reinforcement Learning", Neurocomputing, 2026.

It is deliberately self-contained so it can be added to a BenchMARL checkout
without changing the existing algorithm registry.
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
from torchrl.modules import AdditiveGaussianModule, ProbabilisticActor, TanhDelta

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

# 偏好在网络输入处的重复次数（与 MOMAPPO 的 repeat_interleave(5) 对齐）
PREF_REPEAT = 5

def _make_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    device,
) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers).to(device)


class MultiHeadedActor(nn.Module):
    """A shared feature trunk followed by one action head per agent."""

    def __init__(
        self,
        observation_dim: int,
        preference_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes: Sequence[int],
        device,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("actor_hidden_sizes must contain at least one width")
        self.n_agents = n_agents
        self.action_dim = action_dim
        input_dim = observation_dim + preference_dim * PREF_REPEAT
        feature_dim = int(hidden_sizes[-1])
        self.trunk = _make_mlp(
            input_dim, hidden_sizes[:-1], feature_dim, device=device
        )
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, action_dim, device=device)
            for _ in range(n_agents)
        )

    def forward(
        self, observation: torch.Tensor, preference: torch.Tensor
    ) -> torch.Tensor:
        if observation.shape[-2] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agents, got shape {observation.shape}"
            )
        features = self.trunk(
            torch.cat(
                (observation.to(torch.float32), preference.to(torch.float32).repeat_interleave(PREF_REPEAT, dim=-1)),
                dim=-1,
            )
        )
        return torch.stack(
            [head(features[..., index, :]) for index, head in enumerate(self.heads)],
            dim=-2,
        )


class CentralisedVectorCritic(nn.Module):
    """Centralised critic returning one Q component for each objective."""

    def __init__(
        self,
        central_observation_dim: int,
        joint_action_dim: int,
        preference_dim: int,
        hidden_sizes: Sequence[int],
        device,
    ):
        super().__init__()
        self.network = _make_mlp(
            central_observation_dim + joint_action_dim + preference_dim * PREF_REPEAT,
            hidden_sizes,
            preference_dim,
            device=device,
        )

    def forward(
        self,
        central_observation: torch.Tensor,
        joint_action: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (
                central_observation.flatten(start_dim=-2).to(torch.float32),
                joint_action.flatten(start_dim=-2).to(torch.float32),
                preference.to(torch.float32).repeat_interleave(PREF_REPEAT, dim=-1),
            ),
            dim=-1,
        )
        return self.network(inputs)


class MomaACLoss(nn.Module):
    """Vector Bellman loss and scalarised deterministic policy gradient."""

    def __init__(
        self,
        actor: MultiHeadedActor,
        critic_1: CentralisedVectorCritic,
        critic_2: CentralisedVectorCritic | None,
        group: str,
        n_agents: int,
        n_objectives: int,
        gamma: float,
        polyak_tau: float,
        policy_delay: int,
        target_policy_noise: float,
        target_noise_clip: float,
        loss_function: str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        state_key=None,
    ):
        super().__init__()
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_actor = copy.deepcopy(actor).requires_grad_(False)
        self.target_critic_1 = copy.deepcopy(critic_1).requires_grad_(False)
        self.target_critic_2 = (
            copy.deepcopy(critic_2).requires_grad_(False)
            if critic_2 is not None
            else None
        )
        self.group = group
        self.n_agents = n_agents
        self.n_objectives = n_objectives
        self.gamma = float(gamma)
        self.polyak_tau = float(polyak_tau)
        self.policy_delay = max(int(policy_delay), 1)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.loss_function = loss_function
        self.state_key = state_key
        actor_device = next(actor.parameters()).device
        self.register_buffer(
            "action_low",
            action_low.detach().clone().to(device=actor_device, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            action_high.detach().clone().to(device=actor_device, dtype=torch.float32),
        )
        self.register_buffer(
            "_updates", torch.zeros((), dtype=torch.long, device=actor_device)
        )
        self.last_actor_update = False
        self._pending_target_update = False

    @property
    def online_critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic_1.parameters()
        if self.critic_2 is not None:
            yield from self.critic_2.parameters()

    def _preference(self, td: TensorDictBase) -> torch.Tensor:
        preference = td.get((self.group, "preference")).to(torch.float32)
        if preference.shape[-1] != self.n_objectives:
            raise ValueError(
                f"Expected {self.n_objectives} objectives, got {preference.shape}"
            )
        return preference.mean(dim=-2)

    def _central_observation(self, td: TensorDictBase) -> torch.Tensor:
        if self.state_key is not None:
            state = td.get(self.state_key)
            return state.reshape(*td.batch_size, -1).unsqueeze(-2)
        return td.get((self.group, "observation"))

    def _actor_action(
        self,
        actor: MultiHeadedActor,
        observation: torch.Tensor,
        preference_by_agent: torch.Tensor,
    ) -> torch.Tensor:
        raw_action = actor(observation, preference_by_agent)
        unit_action = raw_action.tanh()
        return self.action_low + (unit_action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

    def _vector_reward(self, next_td: TensorDictBase) -> torch.Tensor:
        candidates = (
            (self.group, "reward_vector"),
            (self.group, "vector_reward"),
            (self.group, "mo_reward"),
            "reward_vector",
            "vector_reward",
            "mo_reward",
        )
        for key in candidates:
            if key not in next_td.keys(True, True):
                continue
            reward = next_td.get(key).to(torch.float32)
            if reward.shape[-1] != self.n_objectives:
                continue
            if reward.ndim >= 2 and reward.shape[-2] == self.n_agents:
                # MOMA-AC assumes a shared team reward. Mean is identical for a
                # truly shared reward and is robust to replicated agent entries.
                reward = reward.mean(dim=-2)
            return reward
        raise KeyError(
            "MOMA-AC requires a vector reward under reward_vector, "
            "vector_reward, or mo_reward (top-level or inside the agent group)."
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
        raise KeyError("MOMA-AC could not find done/terminated in the batch")

    def _distance(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_function == "l1":
            return F.l1_loss(prediction, target)
        if self.loss_function == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        if self.loss_function == "l2":
            return F.mse_loss(prediction, target)
        raise ValueError(f"Unknown loss_function: {self.loss_function}")

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        pairs = (
            (self.target_actor, self.actor),
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2),
        )
        for target, source in pairs:
            if target is None:
                continue
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.lerp_(source_param, self.polyak_tau)

    def _critic_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = self._central_observation(td)
        action = td.get((self.group, "action"))
        q_1 = self.critic_1(observation, action, preference)
        q_2 = (
            self.critic_2(observation, action, preference)
            if self.critic_2 is not None
            else None
        )

        with torch.no_grad():
            next_td = td.get("next")
            next_observation_local = next_td.get((self.group, "observation"))
            preference_by_agent = td.get((self.group, "preference"))
            next_action = self._actor_action(
                self.target_actor, next_observation_local, preference_by_agent
            )
            if self.target_critic_2 is not None:
                noise = torch.randn_like(next_action) * self.target_policy_noise
                noise = noise.clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_action = (next_action + noise).clamp(
                    self.action_low, self.action_high
                )

            next_observation = self._central_observation(next_td)
            target_q_1 = self.target_critic_1(
                next_observation, next_action, preference
            )
            if self.target_critic_2 is None:
                bootstrap_vector = target_q_1
            else:
                target_q_2 = self.target_critic_2(
                    next_observation, next_action, preference
                )
                utility_1 = (target_q_1 * preference).sum(dim=-1, keepdim=True)
                utility_2 = (target_q_2 * preference).sum(dim=-1, keepdim=True)
                # Equation 5/6: select the entire Q vector belonging to the
                # lower scalarised utility. This is not componentwise min.
                bootstrap_vector = torch.where(
                    utility_1 <= utility_2, target_q_1, target_q_2
                )

            reward = self._vector_reward(next_td)
            done = self._done(next_td)
            while done.ndim < reward.ndim:
                done = done.unsqueeze(-1)
            target = reward + self.gamma * (1.0 - done) * bootstrap_vector

        loss = self._distance(q_1, target)
        if q_2 is not None:
            loss = loss + self._distance(q_2, target)
        td_error = (target - q_1.detach()).norm(dim=-1, keepdim=True)
        mean_utility = (q_1.detach() * preference).sum(dim=-1).mean()
        return loss, td_error, mean_utility

    def _actor_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> torch.Tensor:
        local_observation = td.get((self.group, "observation"))
        preference_by_agent = td.get((self.group, "preference"))
        policy_actions = self._actor_action(
            self.actor, local_observation, preference_by_agent
        )
        replay_actions = td.get((self.group, "action"))
        central_observation = self._central_observation(td)

        critic_parameters = list(self.critic_1.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            losses = []
            for agent_index in range(self.n_agents):
                joint_action = replay_actions.clone()
                joint_action[..., agent_index, :] = policy_actions[
                    ..., agent_index, :
                ]
                q_vector = self.critic_1(
                    central_observation, joint_action, preference
                )
                losses.append(-(q_vector * preference).sum(dim=-1).mean())
            return torch.stack(losses).mean()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        if self._pending_target_update:
            self._soft_update_targets()
            self._pending_target_update = False

        self._updates += 1
        self.last_actor_update = (
            int(self._updates.item()) % self.policy_delay == 0
        )
        preference = self._preference(tensordict)
        critic_loss, td_error, mean_utility = self._critic_loss(
            tensordict, preference
        )

        if self.last_actor_update:
            actor_loss = self._actor_loss(tensordict, preference)
            self._pending_target_update = True
        else:
            with torch.no_grad():
                actor_loss = torch.zeros((), device=critic_loss.device)

        priority = td_error.unsqueeze(-2).expand(
            *td_error.shape[:-1], self.n_agents, 1
        )
        tensordict.set((self.group, "td_error"), priority)
        return TensorDict(
            {
                # Actor comes first: BenchMARL steps optimizers in output order.
                "loss_actor": actor_loss,
                "loss_critic": critic_loss,
                "td_error": td_error.mean(),
                "utility": mean_utility,
                # BenchMARL adds this key only when the delayed actor optimizer
                # runs. Keep a placeholder on skipped TD3 steps so the training
                # TensorDicts always have an identical set of keys and can be
                # stacked for logging. On actor-update steps, the experiment
                # loop overwrites it with the measured gradient norm.
                "grad_norm_loss_actor": actor_loss.new_zeros(()),
            },
            batch_size=[],
            device=critic_loss.device,
        )


class MomaAC(Algorithm):
    """MOMA-TD3/MOMA-DDPG under BenchMARL's off-policy training loop."""

    def __init__(
        self,
        variant: str,
        actor_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        loss_function: str,
        policy_delay: int,
        target_policy_noise: float,
        target_noise_clip: float,
        exploration_noise: float,
        exploration_noise_end: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        variant = variant.lower()
        if variant not in {"td3", "ddpg"}:
            raise ValueError("variant must be 'td3' or 'ddpg'")
        self.variant = variant
        self.actor_hidden_sizes = tuple(actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(critic_hidden_sizes)
        self.loss_function = loss_function
        self.policy_delay = policy_delay if variant == "td3" else 1
        self.target_policy_noise = target_policy_noise if variant == "td3" else 0.0
        self.target_noise_clip = target_noise_clip if variant == "td3" else 0.0
        self.exploration_noise = exploration_noise
        self.exploration_noise_end = exploration_noise_end
        self._actors: Dict[str, MultiHeadedActor] = {}

    def _dimensions(self, group: str):
        required = ((group, "observation"), (group, "preference"))
        for key in required:
            if key not in self.observation_spec.keys(True, True):
                raise KeyError(
                    f"MOMA-AC requires observation key {key}; "
                    "enable condition_on_preference in the task."
                )
        n_agents = len(self.group_map[group])
        observation_shape = self.observation_spec[group, "observation"].shape
        preference_shape = self.observation_spec[group, "preference"].shape
        action_shape = self.action_spec[group, "action"].shape
        if any(len(shape) != 2 for shape in (
            observation_shape,
            preference_shape,
            action_shape,
        )):
            raise ValueError(
                "MOMA-AC expects vector observations, preferences, and actions "
                "with specs (n_agents, feature_dim). Zero-pad heterogeneous "
                "local observations as prescribed by the paper."
            )
        observation_dim = int(observation_shape[-1])
        preference_dim = int(preference_shape[-1])
        action_dim = int(action_shape[-1])
        if self.state_spec is None:
            central_observation_dim = n_agents * observation_dim
            state_key = None
        else:
            state_key = list(self.state_spec.keys(True, True))[0]
            central_observation_dim = int(
                torch.tensor(self.state_spec[state_key].shape).prod().item()
            )
        return (
            n_agents,
            observation_dim,
            preference_dim,
            action_dim,
            central_observation_dim,
            state_key,
        )

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("MOMA-AC supports continuous actions only")
        (
            n_agents,
            observation_dim,
            preference_dim,
            action_dim,
            _,
            _,
        ) = self._dimensions(group)
        actor = MultiHeadedActor(
            observation_dim=observation_dim,
            preference_dim=preference_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_sizes=self.actor_hidden_sizes,
            device=self.device,
        )
        self._actors[group] = actor
        actor_module = TensorDictModule(
            actor,
            in_keys=[(group, "observation"), (group, "preference")],
            out_keys=[(group, "param")],
        )
        return ProbabilisticActor(
            module=actor_module,
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": self.action_spec[group, "action"].space.low,
                "high": self.action_spec[group, "action"].space.high,
            },
            return_log_prob=False,
            safe=False,
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("MOMA-AC supports continuous actions only")
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
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[MomaACLoss, bool]:
        if not continuous:
            raise NotImplementedError("MOMA-AC supports continuous actions only")
        (
            n_agents,
            _,
            n_objectives,
            action_dim,
            central_observation_dim,
            state_key,
        ) = self._dimensions(group)
        critic_args = dict(
            central_observation_dim=central_observation_dim,
            joint_action_dim=n_agents * action_dim,
            preference_dim=n_objectives,
            hidden_sizes=self.critic_hidden_sizes,
            device=self.device,
        )
        critic_1 = CentralisedVectorCritic(**critic_args)
        critic_2 = (
            CentralisedVectorCritic(**critic_args)
            if self.variant == "td3"
            else None
        )
        action_spec = self.action_spec[group, "action"]
        loss = MomaACLoss(
            actor=self._actors[group],
            critic_1=critic_1,
            critic_2=critic_2,
            group=group,
            n_agents=n_agents,
            n_objectives=n_objectives,
            gamma=self.experiment_config.gamma,
            polyak_tau=self.experiment_config.polyak_tau,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            loss_function=self.loss_function,
            action_low=action_spec.space.low,
            action_high=action_spec.space.high,
            state_key=state_key,
        ).to(self.device)
        # Targets are updated internally at the delayed actor-update cadence.
        return loss, False

    def _get_parameters(
        self, group: str, loss: MomaACLoss
    ) -> Dict[str, Iterable]:
        return {
            "loss_actor": loss.actor.parameters(),
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
        # Vector rewards and preferences are kept intact; no pre-scalarisation.
        return batch


@dataclass
class MomaACConfig(AlgorithmConfig):
    """Configuration for the paper's MOMA-TD3 or MOMA-DDPG instance."""

    variant: str = MISSING
    actor_hidden_sizes: List[int] = MISSING
    critic_hidden_sizes: List[int] = MISSING
    loss_function: str = MISSING
    policy_delay: int = MISSING
    target_policy_noise: float = MISSING
    target_noise_clip: float = MISSING
    exploration_noise: float = MISSING
    exploration_noise_end: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MomaAC

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
