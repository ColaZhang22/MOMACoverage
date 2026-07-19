from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import IndependentNormal, MaskedCategorical, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.mappo import Mappo
from benchmarl.models.common import ModelConfig


class CMOMappo(Mappo):
    """Preference-conditioned MAPPO.

    输入键:
      - (group, "observation"): 向量观测
      - (group, "preference"):  偏好权重 w (K 维), 每个 agent 都带一份
    融合成:
      - (group, "fused_observation") = concat(observation, preference)
    奖励: 由环境用 w 标量化后写入 (group, "reward"), GAE 不变.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    # ---- 维度: obs_dim + K * pref_repeat_factor ----
    def _get_fused_obs_dim(self, group: str) -> int:
        if (group, "observation") not in self.observation_spec.keys(True, True):
            raise KeyError(
                f"CMOMappo expects observation key {(group, 'observation')}, "
                f"got {list(self.observation_spec.keys(True, True))}."
            )
        if (group, "preference") not in self.observation_spec.keys(True, True):
            raise KeyError(
                f"CMOMappo expects preference key {(group, 'preference')}. "
                "Enable condition_on_preference in the task config or use MAPPO."
            )
        obs_shape = self.observation_spec[group, "observation"].shape
        pref_shape = self.observation_spec[group, "preference"].shape
        return int(obs_shape[-1] + pref_shape[-1] * 5)

    # ---- 融合器: concat ----
    def _get_obs_fuser(self, group: str) -> TensorDictModule:
        def fuse_fn(obs: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
            # print(f"obs shape: {obs.shape}, pref shape: {pref.shape}")
            pref_repeat = pref.repeat_interleave(5, dim=-1)
            # print(f"pref_repeat shape: {pref_repeat.shape}")
            return torch.cat([obs.to(torch.float32), pref_repeat.to(torch.float32)], dim=-1)

        return TensorDictModule(
            fuse_fn,
            in_keys=[(group, "observation"), (group, "preference")],
            out_keys=[(group, "fused_observation")],
        )

    def _get_fused_input_spec(self, group: str) -> Composite:
        n_agents = len(self.group_map[group])
        fused_dim = self._get_fused_obs_dim(group)
        return Composite(
            {
                group: Composite(
                    {
                        "fused_observation": Unbounded(
                            shape=(n_agents, fused_dim),
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )

    # ---- loss 与父类一致 ----
    def _get_loss(self, group, policy_for_loss, continuous) -> Tuple[LossModule, bool]:
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group, loss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    # ---- actor: 输入用 fused_observation ----
    def _get_policy_for_loss(self, group, model_config, continuous) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = self._get_fused_input_spec(group)
        actor_output_spec = Composite(
            {group: Composite(
                {"logits": Unbounded(shape=logits_shape)}, shape=(n_agents,)
            )}
        )

        fuser = self._get_obs_fuser(group)
        actor_core = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        actor_module = TensorDictSequential(fuser, actor_core)

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=IndependentNormal if not self.use_tanh_normal else TanhNormal,
                distribution_kwargs=(
                    {"low": self.action_spec[(group, "action")].space.low,
                     "high": self.action_spec[(group, "action")].space.high}
                    if self.use_tanh_normal else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        return policy

    # ---- critic: 输入也用 fused_observation ----
    def get_critic(self, group: str) -> TensorDictModule:
        if self.state_spec is not None:
            return super().get_critic(group)  # 有全局 state 时走父类

        n_agents = len(self.group_map[group])

        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {group: Composite(
                    {"state_value": Unbounded(shape=(n_agents, 1))}, shape=(n_agents,)
                )}
            )

        critic_input_spec = self._get_fused_input_spec(group)

        fuser = self._get_obs_fuser(group)
        critic_core = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )
        value_module = TensorDictSequential(fuser, critic_core)

        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(*value.shape[:-1], n_agents, 1),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module


@dataclass
class CMOMappoConfig(AlgorithmConfig):
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return CMOMappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
