#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from dataclasses import dataclass, MISSING
from math import sqrt
from typing import Callable, Dict, Iterable, Optional, Tuple, Type

import torch
from torch import nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase, Transform, TransformedEnv
from torchrl.modules import EGreedyModule
from torchrl.objectives import LossModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.qmix import Qmix
from benchmarl.models.common import ModelConfig


class DiscreteToContinuousAction(Transform):
    """Map one discrete action index to one 2D continuous force vector."""

    def __init__(
        self,
        group: str,
        n_agents: int,
        n_actions: int,
        action_dim: int,
        action_scale: float,
    ):
        super().__init__(
            in_keys=[],
            out_keys=[],
            in_keys_inv=[(group, "action")],
            out_keys_inv=[(group, "action")],
        )
        if n_actions != 8:
            raise ValueError("DiscreteToContinuousAction currently supports 8 actions.")
        if action_dim != 2:
            raise ValueError(
                "MO-MIX action discretization expects a 2D continuous action."
            )
        diag = 1.0 / sqrt(2.0)
        self.group = group
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.action_scale = float(action_scale)
        self.register_buffer(
            "action_table",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [diag, diag],
                    [0.0, 1.0],
                    [-diag, diag],
                    [-1.0, 0.0],
                    [-diag, -diag],
                    [0.0, -1.0],
                    [diag, -diag],
                ],
                dtype=torch.float32,
            )
            * self.action_scale,
        )

    def transform_action_spec(self, action_spec):
        action_spec = action_spec.clone()
        continuous_action_spec = action_spec[self.group, "action"]
        n_agents = continuous_action_spec.shape[-2]
        action_spec[self.group].set(
            "action",
            Categorical(
                n=self.n_actions,
                shape=(n_agents,),
                device=continuous_action_spec.device,
                dtype=torch.long,
            ),
        )
        return action_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action_index = action.to(torch.long)
        if (
            action_index.ndim >= 2
            and action_index.shape[-1] == 1
            and action_index.shape[-2] == self.n_agents
        ):
            action_index = action_index.squeeze(-1)
        action_index = action_index.clamp(0, self.n_actions - 1)
        table = self.action_table.to(device=action.device)
        return table[action_index]


class MultiObjectiveMixer(nn.Module):
    """QMIX-style mixer with one independent parallel track per objective."""

    def __init__(
        self,
        state_shape,
        mixing_embed_dim: int,
        n_agents: int,
        n_objectives: int,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_objectives = n_objectives
        self.embed_dim = mixing_embed_dim
        self.state_ndim = len(state_shape)
        self.state_dim = int(torch.tensor(state_shape).prod().item())

        self.hyper_w_1 = nn.ModuleList(
            [nn.Linear(self.state_dim, n_agents * mixing_embed_dim) for _ in range(n_objectives)]
        )
        self.hyper_b_1 = nn.ModuleList(
            [nn.Linear(self.state_dim, mixing_embed_dim) for _ in range(n_objectives)]
        )
        self.hyper_w_2 = nn.ModuleList(
            [nn.Linear(self.state_dim, mixing_embed_dim) for _ in range(n_objectives)]
        )
        self.hyper_b_2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, mixing_embed_dim),
                    nn.ReLU(),
                    nn.Linear(mixing_embed_dim, 1),
                )
                for _ in range(n_objectives)
            ]
        )
        self.to(device)

    def forward(self, local_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state = state.reshape(*state.shape[: -self.state_ndim], -1)
        local_q = local_q.to(torch.float32)
        state = state.to(torch.float32)
        outs = []
        for objective in range(self.n_objectives):
            q = local_q[..., :, objective].unsqueeze(-2)
            w1 = torch.abs(self.hyper_w_1[objective](state)).view(
                *state.shape[:-1], self.n_agents, self.embed_dim
            )
            b1 = self.hyper_b_1[objective](state).view(
                *state.shape[:-1], 1, self.embed_dim
            )
            hidden = torch.nn.functional.elu(torch.matmul(q, w1) + b1)
            w2 = torch.abs(self.hyper_w_2[objective](state)).view(
                *state.shape[:-1], self.embed_dim, 1
            )
            b2 = self.hyper_b_2[objective](state).view(*state.shape[:-1], 1, 1)
            outs.append((torch.matmul(hidden, w2) + b2).squeeze(-1).squeeze(-1))
        return torch.stack(outs, dim=-1)


class MOMixLoss(nn.Module):
    """TD loss for MO-MIX, with vector-reward support when available."""

    def __init__(
        self,
        policy: TensorDictSequential,
        mixer: TensorDictModule,
        group: str,
        gamma: float,
        loss_function: str,
        delay_value: bool,
        target_update_interval: int,
        soft_target_update: bool,
        polyak_tau: float,
        n_agents: int,
        n_objectives: int,
    ):
        super().__init__()
        self.policy = policy
        self.mixer = mixer
        self.group = group
        self.gamma = gamma
        self.loss_function = loss_function
        self.delay_value = delay_value
        self.target_update_interval = max(int(target_update_interval), 1)
        self.soft_target_update = soft_target_update
        self.polyak_tau = polyak_tau
        self.n_agents = n_agents
        self.n_objectives = n_objectives
        self.register_buffer("_updates", torch.zeros((), dtype=torch.long))

        self.target_policy = copy.deepcopy(policy)
        self.target_mixer = copy.deepcopy(mixer)
        for module in (self.target_policy, self.target_mixer):
            module.requires_grad_(False)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_function == "l1":
            return torch.nn.functional.l1_loss(prediction, target)
        if self.loss_function == "smooth_l1":
            return torch.nn.functional.smooth_l1_loss(prediction, target)
        if self.loss_function != "l2":
            raise ValueError(f"Unknown loss_function: {self.loss_function}")
        return torch.nn.functional.mse_loss(prediction, target)

    def _hard_update_targets(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for target, source in zip(
                self.target_policy.parameters(), self.policy.parameters()
            ):
                target.data.lerp_(source.data, self.polyak_tau)
            for target, source in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target.data.lerp_(source.data, self.polyak_tau)

    def _update_targets(self) -> None:
        if not self.delay_value:
            return
        self._updates += 1
        if self.soft_target_update:
            self._soft_update_targets()
        elif int(self._updates.item()) % self.target_update_interval == 0:
            self._hard_update_targets()

    def _group_preference(self, tensordict: TensorDictBase) -> torch.Tensor:
        preference = tensordict.get((self.group, "preference"))
        return preference.mean(dim=-2)

    def _gather_action_value(
        self, action_value_vector: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        action = action.to(torch.long)
        if (
            action.ndim >= 2
            and action.shape[-1] == 1
            and action.shape[-2] == self.n_agents
        ):
            action = action.squeeze(-1)
        gather_index = action.unsqueeze(-1).unsqueeze(-1).expand(
            *action.shape, 1, action_value_vector.shape[-1]
        )
        return action_value_vector.gather(-2, gather_index).squeeze(-2)

    def _reward_vector(self, tensordict: TensorDictBase) -> Optional[torch.Tensor]:
        candidate_keys = [
            (self.group, "reward_vector"),
            (self.group, "mo_reward"),
            (self.group, "vector_reward"),
            "reward_vector",
            "mo_reward",
            "vector_reward",
        ]
        for key in candidate_keys:
            if key in tensordict.keys(True, True):
                reward = tensordict.get(key).to(torch.float32)
                if (
                    reward.ndim >= 2
                    and reward.shape[-1] == self.n_objectives
                    and reward.shape[-2] == self.n_agents
                ):
                    reward = reward.mean(dim=-2)
                return reward
        return None

    def _scalar_reward(self, tensordict: TensorDictBase) -> torch.Tensor:
        if "reward" in tensordict.keys(True, True):
            return tensordict.get("reward").to(torch.float32)
        return tensordict.get((self.group, "reward")).mean(dim=-2).to(torch.float32)

    def _done(self, tensordict: TensorDictBase) -> torch.Tensor:
        if "terminated" in tensordict.keys(True, True):
            done = tensordict.get("terminated")
        elif "done" in tensordict.keys(True, True):
            done = tensordict.get("done")
        else:
            done = tensordict.get((self.group, "terminated")).any(dim=-2)
        return done.to(torch.float32)

    def _mixed_value(
        self,
        tensordict: TensorDictBase,
        policy: TensorDictSequential,
        mixer: TensorDictModule,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        td = policy(tensordict)
        if action is None:
            local_q = td.get((self.group, "chosen_action_value_vector"))
        else:
            local_q = self._gather_action_value(
                td.get((self.group, "action_value_vector")), action
            )
            td.set((self.group, "chosen_action_value_vector"), local_q)
        td = mixer(td)
        return td.get("chosen_action_value_vector"), td

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        action = tensordict.get((self.group, "action"))
        current_value_vector, _ = self._mixed_value(
            tensordict.clone(False), self.policy, self.mixer, action=action
        )
        preference = self._group_preference(tensordict)
        current_scalar = (current_value_vector * preference).sum(dim=-1, keepdim=True)

        with torch.no_grad():
            next_td = tensordict.get("next").clone(False)
            target_policy = self.target_policy if self.delay_value else self.policy
            target_mixer = self.target_mixer if self.delay_value else self.mixer
            target_value_vector, _ = self._mixed_value(
                next_td, target_policy, target_mixer
            )
            done = self._done(next_td)
            vector_reward = self._reward_vector(next_td)
            if vector_reward is not None:
                target_value = vector_reward + self.gamma * (1.0 - done) * target_value_vector
                td_error = (
                    target_value.detach() - current_value_vector.detach()
                ).norm(dim=-1, keepdim=True)
                loss = self._loss(current_value_vector, target_value)
            else:
                reward = self._scalar_reward(next_td)
                next_preference = self._group_preference(next_td)
                target_scalar = (
                    target_value_vector * next_preference
                ).sum(dim=-1, keepdim=True)
                target_value = reward + self.gamma * (1.0 - done) * target_scalar
                td_error = (target_value.detach() - current_scalar.detach()).abs()
                loss = self._loss(current_scalar, target_value)

        tensordict.set((self.group, "td_error"), td_error)
        self._update_targets()
        return TensorDict(
            {
                "loss": loss,
                "td_error": td_error.mean(),
                "q_value": current_scalar.mean(),
            },
            batch_size=[],
            device=loss.device,
        )


class MOMix(Qmix):
    """Preference-conditioned MO-MIX for discrete actions.

    The agent network outputs one Q-value per objective and action. Actions are
    selected by scalarizing those vectors with ``preference``. The centralized
    mixer is a MOMN-style parallel mixer with one QMIX track per objective.
    """

    def __init__(
        self,
        mixing_embed_dim: int,
        delay_value: bool,
        loss_function: str,
        n_discrete_actions: int,
        action_scale: float,
        preference_repeat_factor: int,
        discretize_continuous_actions: bool,
        **kwargs,
    ):
        super().__init__(
            mixing_embed_dim=mixing_embed_dim,
            delay_value=delay_value,
            loss_function=loss_function,
            **kwargs,
        )
        self.n_discrete_actions = n_discrete_actions
        self.action_scale = action_scale
        self.preference_repeat_factor = preference_repeat_factor
        self.discretize_continuous_actions = discretize_continuous_actions
        if self.discretize_continuous_actions:
            self._replace_continuous_action_specs()

    def _replace_continuous_action_specs(self) -> None:
        self._original_action_spec = self.action_spec.clone()
        for group in self.group_map.keys():
            action_spec = self.action_spec[group, "action"]
            if isinstance(action_spec, Categorical):
                continue
            if len(action_spec.shape) < 2:
                raise ValueError(
                    "MO-MIX expected action spec shape (..., n_agents, action_dim)."
                )
            n_agents, action_dim = action_spec.shape[-2], action_spec.shape[-1]
            if action_dim != 2:
                raise ValueError(
                    f"MO-MIX can discretize only 2D actions, got {action_dim}."
                )
            self.action_spec[group].set(
                "action",
                Categorical(
                    n=self.n_discrete_actions,
                    shape=(n_agents,),
                    device=action_spec.device,
                    dtype=torch.long,
                ),
            )
        self.experiment.action_spec = self.action_spec

    def process_env_fun(
        self,
        env_fun: Callable[[], EnvBase],
    ) -> Callable[[], EnvBase]:
        if not self.discretize_continuous_actions:
            return env_fun

        def make_env() -> EnvBase:
            env = env_fun()
            for group in self.group_map.keys():
                original_action_spec = self._original_action_spec[group, "action"]
                env = TransformedEnv(
                    env,
                    DiscreteToContinuousAction(
                        group=group,
                        n_agents=original_action_spec.shape[-2],
                        n_actions=self.n_discrete_actions,
                        action_dim=original_action_spec.shape[-1],
                        action_scale=self.action_scale,
                    ),
                )
            return env

        return make_env

    def _get_fused_obs_dim(self, group: str) -> int:
        if (group, "observation") not in self.observation_spec.keys(True, True):
            raise KeyError(
                f"MOMix expects observation key {(group, 'observation')}, "
                f"got {list(self.observation_spec.keys(True, True))}."
            )
        if (group, "preference") not in self.observation_spec.keys(True, True):
            raise KeyError(
                f"MOMix expects preference key {(group, 'preference')}. "
                "Enable condition_on_preference in the task config."
            )
        obs_shape = self.observation_spec[group, "observation"].shape
        pref_shape = self.observation_spec[group, "preference"].shape
        return int(obs_shape[-1] + pref_shape[-1] * self.preference_repeat_factor)

    def _get_obs_fuser(self, group: str) -> TensorDictModule:
        def fuse_fn(obs: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
            pref = pref.repeat_interleave(self.preference_repeat_factor, dim=-1)
            return torch.cat([obs.to(torch.float32), pref.to(torch.float32)], dim=-1)

        return TensorDictModule(
            fuse_fn,
            in_keys=[(group, "observation"), (group, "preference")],
            out_keys=[(group, "fused_observation")],
        )

    def _get_fused_input_spec(self, group: str) -> Composite:
        n_agents = len(self.group_map[group])
        return Composite(
            {
                group: Composite(
                    {
                        "fused_observation": Unbounded(
                            shape=(n_agents, self._get_fused_obs_dim(group)),
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            raise NotImplementedError("MO-MIX is not compatible with continuous actions.")

        n_agents = len(self.group_map[group])
        n_objectives = self._get_n_objectives(group)
        logits_shape = [
            *self.action_spec[group, "action"].shape,
            self.action_spec[group, "action"].space.n,
            n_objectives,
        ]
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"action_value_vector": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=self._get_fused_input_spec(group),
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        return TensorDictSequential(
            self._get_obs_fuser(group),
            actor_module,
            self._get_action_selector(group),
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            raise NotImplementedError("MO-MIX is not compatible with continuous actions.")
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
            device=self.device,
        )
        return TensorDictSequential(*policy_for_loss, greedy)

    def _get_action_selector(self, group: str) -> TensorDictModule:
        def select_action(
            action_value_vector: torch.Tensor, preference: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            scalar_action_value = (
                action_value_vector * preference.unsqueeze(-2)
            ).sum(dim=-1)
            action = scalar_action_value.argmax(dim=-1)
            gather_index = action.unsqueeze(-1).unsqueeze(-1).expand(
                *action.shape, 1, action_value_vector.shape[-1]
            )
            chosen_vector = action_value_vector.gather(-2, gather_index).squeeze(-2)
            chosen_scalar = scalar_action_value.gather(
                -1, action.unsqueeze(-1)
            ).squeeze(-1)
            return action, scalar_action_value, chosen_vector, chosen_scalar

        return TensorDictModule(
            select_action,
            in_keys=[(group, "action_value_vector"), (group, "preference")],
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value_vector"),
                (group, "chosen_action_value"),
            ],
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("MO-MIX is not compatible with continuous actions.")
        n_agents = len(self.group_map[group])
        loss = MOMixLoss(
            policy=policy_for_loss,
            mixer=self.get_mixer(group),
            group=group,
            gamma=self.experiment_config.gamma,
            loss_function=self.loss_function,
            delay_value=self.delay_value,
            target_update_interval=self.experiment_config.hard_target_update_frequency,
            soft_target_update=self.experiment_config.soft_target_update,
            polyak_tau=self.experiment_config.polyak_tau,
            n_agents=n_agents,
            n_objectives=self._get_n_objectives(group),
        )
        return loss, False

    def get_mixer(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        n_objectives = self._get_n_objectives(group)
        if self.state_spec is not None:
            global_state_key = list(self.state_spec.keys(True, True))[0]
            state_shape = self.state_spec[global_state_key].shape
            in_keys = [(group, "chosen_action_value_vector"), global_state_key]
        else:
            state_shape = (n_agents, self._get_fused_obs_dim(group))
            in_keys = [
                (group, "chosen_action_value_vector"),
                (group, "fused_observation"),
            ]

        mixer = TensorDictModule(
            module=MultiObjectiveMixer(
                state_shape=state_shape,
                mixing_embed_dim=self.mixing_embed_dim,
                n_agents=n_agents,
                n_objectives=n_objectives,
                device=self.device,
            ),
            in_keys=in_keys,
            out_keys=["chosen_action_value_vector"],
        )
        return mixer

    def _get_n_objectives(self, group: str) -> int:
        if (group, "preference") not in self.observation_spec.keys(True, True):
            raise KeyError(
                f"MOMix expects preference key {(group, 'preference')}. "
                "Enable condition_on_preference in the task config."
            )
        return int(self.observation_spec[group, "preference"].shape[-1])

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {"loss": loss.parameters()}

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        return super().process_batch(group, batch)


@dataclass
class MOMixConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.MOMix`."""

    mixing_embed_dim: int = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING
    n_discrete_actions: int = MISSING
    action_scale: float = MISSING
    preference_repeat_factor: int = MISSING
    discretize_continuous_actions: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MOMix

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
