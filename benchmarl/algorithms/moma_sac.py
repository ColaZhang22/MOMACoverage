"""Preference-conditioned multi-objective multi-agent Soft Actor-Critic.

This is an SAC extension of MOMA-AC.  It keeps MOMA-AC's centralised
vector-valued critics and decentralised preference-conditioned actors, while
replacing the deterministic TD3/DDPG actor with a tanh-Gaussian policy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import MISSING, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.moma_ac import (
    PREF_REPEAT,
    CentralisedVectorCritic,
    _make_mlp,
)
from benchmarl.models.common import ModelConfig


class MultiHeadedGaussianActor(nn.Module):
    """Shared local feature trunk with one Gaussian action head per agent."""

    def __init__(
        self,
        observation_dim: int,
        preference_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        device,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("actor_hidden_sizes must contain at least one width")
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        input_dim = observation_dim + preference_dim * PREF_REPEAT
        feature_dim = int(hidden_sizes[-1])
        self.trunk = _make_mlp(
            input_dim, hidden_sizes[:-1], feature_dim, device=device
        )
        self.loc_heads = nn.ModuleList(
            nn.Linear(feature_dim, action_dim, device=device)
            for _ in range(n_agents)
        )
        self.log_std_heads = nn.ModuleList(
            nn.Linear(feature_dim, action_dim, device=device)
            for _ in range(n_agents)
        )

    def forward(
        self, observation: torch.Tensor, preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if observation.shape[-2] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agents, got shape {observation.shape}"
            )
        features = self.trunk(
            torch.cat(
                (
                    observation.to(torch.float32),
                    preference.to(torch.float32).repeat_interleave(
                        PREF_REPEAT, dim=-1
                    ),
                ),
                dim=-1,
            )
        )
        loc = torch.stack(
            [
                head(features[..., index, :])
                for index, head in enumerate(self.loc_heads)
            ],
            dim=-2,
        )
        log_std = torch.stack(
            [
                head(features[..., index, :])
                for index, head in enumerate(self.log_std_heads)
            ],
            dim=-2,
        ).clamp(self.log_std_min, self.log_std_max)
        return loc, log_std.exp()


class MomaSACLoss(nn.Module):
    """Vector Bellman loss with a scalarised maximum-entropy actor objective."""

    def __init__(
        self,
        actor: MultiHeadedGaussianActor,
        critic_1: CentralisedVectorCritic,
        critic_2: CentralisedVectorCritic,
        group: str,
        n_agents: int,
        n_objectives: int,
        action_dim: int,
        gamma: float,
        polyak_tau: float,
        loss_function: str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        alpha_init: float,
        target_entropy: Union[float, str],
        fixed_alpha: bool,
        min_alpha: Optional[float],
        max_alpha: Optional[float],
        state_key=None,
    ):
        super().__init__()
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        if target_entropy != "auto" and not isinstance(
            target_entropy, (float, int)
        ):
            raise ValueError("target_entropy must be a number or 'auto'")

        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = copy.deepcopy(critic_1).requires_grad_(False)
        self.target_critic_2 = copy.deepcopy(critic_2).requires_grad_(False)
        self.group = group
        self.n_agents = n_agents
        self.n_objectives = n_objectives
        self.gamma = float(gamma)
        self.polyak_tau = float(polyak_tau)
        self.loss_function = loss_function
        self.fixed_alpha = bool(fixed_alpha)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.state_key = state_key
        self.target_entropy = (
            -float(n_agents * action_dim)
            if target_entropy == "auto"
            else float(target_entropy)
        )

        actor_device = next(actor.parameters()).device
        self.register_buffer(
            "action_low",
            action_low.detach().clone().to(
                device=actor_device, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_high",
            action_high.detach().clone().to(
                device=actor_device, dtype=torch.float32
            ),
        )
        initial_log_alpha = torch.tensor(
            math.log(alpha_init), device=actor_device, dtype=torch.float32
        )
        if self.fixed_alpha:
            self.register_buffer("log_alpha", initial_log_alpha)
        else:
            self.log_alpha = nn.Parameter(initial_log_alpha)
        self._pending_target_update = False

    @property
    def online_critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic_1.parameters()
        yield from self.critic_2.parameters()

    @torch.no_grad()
    def _clamp_log_alpha(self) -> None:
        lower = (
            math.log(self.min_alpha)
            if self.min_alpha is not None
            else -float("inf")
        )
        upper = (
            math.log(self.max_alpha)
            if self.max_alpha is not None
            else float("inf")
        )
        self.log_alpha.clamp_(lower, upper)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

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

    def _preference_by_agent(self, td: TensorDictBase) -> torch.Tensor:
        return td.get((self.group, "preference")).to(torch.float32)

    def _sample_action(
        self,
        observation: torch.Tensor,
        preference_by_agent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, scale = self.actor(observation, preference_by_agent)
        distribution = TanhNormal(
            loc,
            scale,
            low=self.action_low,
            high=self.action_high,
            event_dims=1,
        )
        action = distribution.rsample()
        # TanhNormal reduces the action dimension.  The joint policy is the
        # product of decentralised agent policies, hence the sum over agents.
        joint_log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, joint_log_prob

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
                reward = reward.mean(dim=-2)
            return reward
        raise KeyError(
            "MOMA-SAC requires a vector reward under reward_vector, "
            "vector_reward, or mo_reward (top-level or inside the agent group)."
        )

    def _done(self, next_td: TensorDictBase) -> torch.Tensor:
        for key in ("terminated", "done"):
            if key in next_td.keys(True, True):
                return next_td.get(key).to(torch.float32)
        for key in ((self.group, "terminated"), (self.group, "done")):
            if key in next_td.keys(True, True):
                return next_td.get(key).any(dim=-2).to(torch.float32)
        raise KeyError("MOMA-SAC could not find done/terminated in the batch")

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
        for target, source in (
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2),
        ):
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.lerp_(source_param, self.polyak_tau)

    def _critic_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = self._central_observation(td)
        action = td.get((self.group, "action"))
        q_1 = self.critic_1(observation, action, preference)
        q_2 = self.critic_2(observation, action, preference)

        with torch.no_grad():
            next_td = td.get("next")
            next_local_observation = next_td.get(
                (self.group, "observation")
            )
            if (self.group, "preference") in next_td.keys(True, True):
                next_preference_by_agent = self._preference_by_agent(next_td)
            else:
                next_preference_by_agent = self._preference_by_agent(td)
            next_action, next_log_prob = self._sample_action(
                next_local_observation, next_preference_by_agent
            )
            next_observation = self._central_observation(next_td)
            target_q_1 = self.target_critic_1(
                next_observation, next_action, preference
            )
            target_q_2 = self.target_critic_2(
                next_observation, next_action, preference
            )
            utility_1 = (target_q_1 * preference).sum(
                dim=-1, keepdim=True
            )
            utility_2 = (target_q_2 * preference).sum(
                dim=-1, keepdim=True
            )
            selected_q = torch.where(
                utility_1 <= utility_2, target_q_1, target_q_2
            )

            # Broadcasting the joint entropy bonus over objective components
            # preserves the standard soft Bellman equation after scalarisation
            # whenever preference weights sum to one.
            soft_bootstrap = selected_q - self.alpha * next_log_prob.unsqueeze(
                -1
            )
            reward = self._vector_reward(next_td)
            done = self._done(next_td)
            while done.ndim < reward.ndim:
                done = done.unsqueeze(-1)
            target = reward + self.gamma * (1.0 - done) * soft_bootstrap

        loss = self._distance(q_1, target) + self._distance(q_2, target)
        td_error = (target - q_1.detach()).norm(dim=-1, keepdim=True)
        q_utility = (q_1.detach() * preference).sum(dim=-1).mean()
        target_utility = (target * preference).sum(dim=-1).mean()
        return loss, td_error, q_utility, target_utility

    def _actor_and_alpha_loss(
        self, td: TensorDictBase, preference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_observation = td.get((self.group, "observation"))
        preference_by_agent = self._preference_by_agent(td)
        policy_action, joint_log_prob = self._sample_action(
            local_observation, preference_by_agent
        )
        central_observation = self._central_observation(td)

        critic_parameters = list(self.critic_1.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            q_vector = self.critic_1(
                central_observation, policy_action, preference
            )
            utility = (q_vector * preference).sum(dim=-1)
            actor_loss = (
                self.alpha.detach() * joint_log_prob - utility
            ).mean()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

        if self.fixed_alpha:
            alpha_loss = actor_loss.new_zeros(())
        else:
            alpha_loss = -(
                self.log_alpha
                * (joint_log_prob.detach() + self.target_entropy)
            ).mean()
        entropy = -joint_log_prob.detach().mean()
        return actor_loss, alpha_loss, entropy

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        if self._pending_target_update:
            self._soft_update_targets()
        self._pending_target_update = True
        self._clamp_log_alpha()

        preference = self._preference(tensordict)
        critic_loss, td_error, utility, target_utility = self._critic_loss(
            tensordict, preference
        )
        actor_loss, alpha_loss, entropy = self._actor_and_alpha_loss(
            tensordict, preference
        )
        priority = td_error.unsqueeze(-2).expand(
            *td_error.shape[:-1], self.n_agents, 1
        )
        tensordict.set((self.group, "td_error"), priority)

        return TensorDict(
            {
                # Optimisers are stepped in this insertion order.
                "loss_actor": actor_loss,
                "loss_critic": critic_loss,
                "loss_alpha": alpha_loss,
                "td_error": td_error.mean(),
                "utility": utility,
                "target_utility": target_utility,
                "alpha": self.alpha.detach(),
                "entropy": entropy,
            },
            batch_size=[],
            device=critic_loss.device,
        )


class MomaSAC(Algorithm):
    """MOMA-AC with a maximum-entropy SAC backbone."""

    def __init__(
        self,
        actor_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        loss_function: str,
        alpha_init: float,
        target_entropy: Union[float, str],
        fixed_alpha: bool,
        min_alpha: Optional[float],
        max_alpha: Optional[float],
        log_std_min: float,
        log_std_max: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.actor_hidden_sizes = tuple(actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(critic_hidden_sizes)
        self.loss_function = loss_function
        self.alpha_init = alpha_init
        self.target_entropy = target_entropy
        self.fixed_alpha = fixed_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._actors: Dict[str, MultiHeadedGaussianActor] = {}

    def _dimensions(self, group: str):
        required = ((group, "observation"), (group, "preference"))
        for key in required:
            if key not in self.observation_spec.keys(True, True):
                raise KeyError(
                    f"MOMA-SAC requires observation key {key}; "
                    "enable condition_on_preference in the task."
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
                "MOMA-SAC expects vector observations, preferences, and "
                "actions with specs (n_agents, feature_dim)."
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
            raise NotImplementedError("MOMA-SAC supports continuous actions only")
        (
            n_agents,
            observation_dim,
            preference_dim,
            action_dim,
            _,
            _,
        ) = self._dimensions(group)
        actor = MultiHeadedGaussianActor(
            observation_dim=observation_dim,
            preference_dim=preference_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_sizes=self.actor_hidden_sizes,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            device=self.device,
        )
        self._actors[group] = actor
        actor_module = TensorDictModule(
            actor,
            in_keys=[(group, "observation"), (group, "preference")],
            out_keys=[(group, "loc"), (group, "scale")],
        )
        return ProbabilisticActor(
            module=actor_module,
            spec=self.action_spec[group, "action"],
            in_keys=[(group, "loc"), (group, "scale")],
            out_keys=[(group, "action")],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.action_spec[group, "action"].space.low,
                "high": self.action_spec[group, "action"].space.high,
                "event_dims": 1,
            },
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
            safe=False,
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError("MOMA-SAC supports continuous actions only")
        return policy_for_loss

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[MomaSACLoss, bool]:
        if not continuous:
            raise NotImplementedError("MOMA-SAC supports continuous actions only")
        (
            n_agents,
            _,
            n_objectives,
            action_dim,
            central_observation_dim,
            state_key,
        ) = self._dimensions(group)
        critic_args = {
            "central_observation_dim": central_observation_dim,
            "joint_action_dim": n_agents * action_dim,
            "preference_dim": n_objectives,
            "hidden_sizes": self.critic_hidden_sizes,
            "device": self.device,
        }
        action_spec = self.action_spec[group, "action"]
        loss = MomaSACLoss(
            actor=self._actors[group],
            critic_1=CentralisedVectorCritic(**critic_args),
            critic_2=CentralisedVectorCritic(**critic_args),
            group=group,
            n_agents=n_agents,
            n_objectives=n_objectives,
            action_dim=action_dim,
            gamma=self.experiment_config.gamma,
            polyak_tau=self.experiment_config.polyak_tau,
            loss_function=self.loss_function,
            action_low=action_spec.space.low,
            action_high=action_spec.space.high,
            alpha_init=self.alpha_init,
            target_entropy=self.target_entropy,
            fixed_alpha=self.fixed_alpha,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            state_key=state_key,
        ).to(self.device)
        # Target critics are updated internally after each optimiser step.
        return loss, False

    def _get_parameters(
        self, group: str, loss: MomaSACLoss
    ) -> Dict[str, Iterable]:
        parameters = {
            "loss_actor": loss.actor.parameters(),
            "loss_critic": loss.online_critic_parameters,
        }
        if not self.fixed_alpha:
            parameters["loss_alpha"] = [loss.log_alpha]
        return parameters

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        keys = ["loss_actor", "loss_critic"]
        if not self.fixed_alpha:
            keys.append("loss_alpha")
        return loss_vals.select(*keys)

    def process_batch(
        self, group: str, batch: TensorDictBase
    ) -> TensorDictBase:
        return batch


@dataclass
class MomaSACConfig(AlgorithmConfig):
    """Configuration for preference-conditioned vector-valued MOMA-SAC."""

    actor_hidden_sizes: List[int] = MISSING
    critic_hidden_sizes: List[int] = MISSING
    loss_function: str = MISSING
    alpha_init: float = MISSING
    target_entropy: Union[float, str] = MISSING
    fixed_alpha: bool = MISSING
    min_alpha: Optional[float] = MISSING
    max_alpha: Optional[float] = MISSING
    log_std_min: float = MISSING
    log_std_max: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MomaSAC

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
