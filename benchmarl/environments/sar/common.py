import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase
from .sarwrapper import SARWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
# import mss

class SARClass(TaskClass):
    def _resolve_n_agents(self, env: EnvBase) -> int:
        # 1) Native SARWrapper / unwrapped envs
        n_agents = getattr(env, "n_agents", None)
        if isinstance(n_agents, int):
            return n_agents

        # 2) Wrapped envs that already expose group_map
        group_map = getattr(env, "group_map", None)
        if isinstance(group_map, dict) and "agents" in group_map:
            return len(group_map["agents"])

        # 3) Infer from action spec
        for spec_name in ("full_action_spec_unbatched", "full_action_spec"):
            spec = getattr(env, spec_name, None)
            if spec is None:
                continue
            try:
                action_shape = spec["agents", "action"].shape
                if len(action_shape) >= 2:
                    return int(action_shape[-2])
            except Exception:
                pass

        # 4) Unwrap common container attributes (SerialEnv / ParallelEnv / TransformedEnv)
        for attr in ("_dummy_env", "base_env", "env"):
            child_env = getattr(env, attr, None)
            if child_env is not None and child_env is not env:
                try:
                    return self._resolve_n_agents(child_env)
                except Exception:
                    pass

        raise TypeError(
            f"Cannot resolve number of agents for env type {type(env).__name__}. "
            "Expected an integer n_agents or an action spec with ('agents', 'action')."
        )

    def get_env_fun(
        self,
        worker_id: int,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
        env_type: str = "train",
        max_episodes: int = 1,
        adaptive_preference_sampler = None,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        env_path = self._get_env_path_from_task(self.name, config)

        if adaptive_preference_sampler is None:
            print("No preference sampler provided, using truncated normal preference sampling")
        else:
            print("Using preference sampling with adaptive preference sampler")

        return lambda: SARWrapper(
                env_path=env_path,
                worker_id=worker_id,
                scenario=self.name.lower(),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                max_steps=config.get("max_steps", 1000),
                width=config.get("width", 600),
                height=config.get("height", 600),
                time_scale=config.get("time_scale", 10.0),
                render=config.get("render", True),
                manual_metrics_size=config.get("manual_metrics_size", 5),
                condition_on_preference=config.get("condition_on_preference", False),
                preference_alpha=config.get("preference_alpha", 0.0),
                fixed_preference=config.get("fixed_preference", [0.0, 0.0]),
                u_sample_start=config.get("u_sample_start", 0.5),
                env_type=env_type,
                max_episodes=max_episodes,
                adaptive_preference_sampler=adaptive_preference_sampler,
                normalize_mo_reward_rms=config.get("normalize_mo_reward_rms", False),
                debug_save_sensor=config.get("debug_save_sensor", False),
                debug_save_every_n=config.get("debug_save_every_n", 100),
                debug_sensor_dir=config.get("debug_sensor_dir", "debug_sensor_frames"),
            )
    
    def _get_env_path_from_task(self, task_name: str, config: Dict) -> str:
        """根据任务名确定环境路径"""
        # 方式1：从配置文件读取
        if "env_path" in config:
            return config["env_path"]
        
        # 方式2：根据任务名映射路径
        path_mapping = {
            "MAPF": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\MAPF_EASY_MAZE_FOG\UnityEnvironment",
            "EASY_MAZE_FOG": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\EASY_MAZE_FOG_V2\UnityEnvironment",
            "EASY_MAZE_STATIC": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\MAPF_EASY_MAZE\UnityEnvironment",
            # 添加更多任务到路径的映射
        }
        
        return path_mapping.get(task_name, config.get("default_env_path", ""))
    
    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 1000)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        n_agents = self._resolve_n_agents(env)
        return {"agents": [f"agent_{i}" for i in range(n_agents)]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        if hasattr(env, "full_observation_spec_unbatched"):
            return env.full_observation_spec_unbatched
        return env.full_observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        if hasattr(env, "full_action_spec_unbatched"):
            return env.full_action_spec_unbatched
        return env.full_action_spec

    # @staticmethod
    # def log_info(batch: TensorDictBase) -> Dict[str, float]:
    #     to_log: Dict[str, float] = {}
    #     key = ("next", "manual_metrics")
    #     if key in batch.keys(True, True):
    #         manual_metrics = batch.get(key).to(torch.float32)
    #         if manual_metrics.numel() > 0:
    #             # Flatten batch/time/env dims and keep metric dim as the last axis.
    #             flat_metrics = manual_metrics.reshape(-1, manual_metrics.shape[-1])
    #             last_metrics = manual_metrics.reshape(-1, manual_metrics.shape[-1])[-1]

    #             for i in range(flat_metrics.shape[-1]):
    #                 metric = flat_metrics[:, i]
    #                 metric_name = f"collection/manual_metrics/metric_{i + 1}"
    #                 to_log[f"{metric_name}_min"] = metric.min().item()
    #                 to_log[f"{metric_name}_mean"] = metric.mean().item()
    #                 to_log[f"{metric_name}_max"] = metric.max().item()
    #                 to_log[f"{metric_name}_std"] = metric.std(unbiased=False).item()
    #                 to_log[f"{metric_name}_last"] = last_metrics[i].item()

    #             # Aggregate diagnostics across all manual metrics
    #             to_log["collection/manual_metrics/global_mean"] = flat_metrics.mean().item()
    #             to_log["collection/manual_metrics/global_std"] = flat_metrics.std(
    #                 unbiased=False
    #             ).item()
    #     return to_log

    @staticmethod
    def env_name() -> str:
        return "sar"  


class SARTask(Task):
    """Enum for Unity ML-Agents tasks."""
    
    EASY_MAZE_STATIC = None
    EASY_MAZE_FOG = None
    EASY_MAZE_FIRE = None
    EASY_MAZE_STATIC16 = None
    
    MIDDLE_MAZE_STATIC = None
    MIDDLE_MAZE_STATIC16 = None
    MIDDLE_MAZE_FIRE = None
    MIDDLE_MAZE_FOG = None

    MIDDLE_SCHOOL_STATIC = None
    MIDDLE_SCHOOL_FIRE = None
    MIDDLE_SCHOOL_FOG = None
    
    MIDDLE_OFFICE_STATIC = None
    MIDDLE_OFFICE_FIRE = None
    MIDDLE_OFFICE_FOG = None

    HARD_MAZE_STATIC = None
    HARD_MAZE_FIRE = None
    HARD_MAZE_FOG = None

  
    CMSBERLIN = None
    # CMSROOM = None
    # CMSEMPTY = None
    CMSEMPTY180 = None
    CMSEMPTY128 = None
    CMSEMPTY256 = None

    MOEMPTY128 = None
    MOBERLIN128 = None
    MOROOM128 = None
    MOROOM1283A = None
    MOROOM1282A = None

    MOROOM128CPU = None
    MOEMPTY128CPU = None
    MOMAZE128CPU = None
    
    TOWN = None

    @staticmethod
    def associated_class():
        return SARClass
