import torch
import numpy as np
from typing import Optional, List
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuousTensorSpec, BoundedTensorSpec, DiscreteTensorSpec
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import atexit 
from benchmarl.utils import DEVICE_TYPING
from pathlib import Path
import imageio.v2 as imageio
import torch.nn.functional as F
from gymnasium.wrappers.utils import RunningMeanStd


class SARWrapper(EnvBase):
    """Unity ML-Agents environment wrapper for BenchMARL"""
    
    def __init__(
        self,
        env_path: str,
        scenario: str = "easymazefire",
        num_envs: int = 1,
        continuous_actions: bool = True,
        seed: Optional[int] = None,
        device: DEVICE_TYPING = "cpu",
        max_steps: int = 1024,
        worker_id: int = 0, 
        env_type: str = "train",
        max_episodes: int = 1,
        adaptive_preference_sampler = None,
        **kwargs,
    ):  
        print(f"kwargs: {kwargs}")
        self.env_path = env_path
        self.scenario = scenario
        self.num_envs = num_envs  # Unity环境通常是单环境
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.worker_id = worker_id 
        self._step_count = 0
        self.manual_metrics_size = kwargs.get("manual_metrics_size", 4)
        self.env_type = env_type
        self.max_episodes = max_episodes

        # Two-objective problem used by CMOMAPPO and the MOMA family:
        # [exploration gain, path penalty]. The former time-penalty component
        # was constant per step and is intentionally excluded.
        self.mo_reward_dim = 2
        self.condition_on_preference = kwargs.get("condition_on_preference", True)
        self.preference_alpha = kwargs.get("preference_alpha", 1.0)   # Dirichlet parameter
        self.fixed_preference = kwargs.get("fixed_preference", [1.0, 0.0])  # Fixed preference for evaluation
        self.u_sample_start = kwargs.get("u_sample_start", 0.5)
        self.current_w = np.array([1.0, 0.0], dtype=np.float32)  # Placeholder
        
        # Multi-ojectives Reward Normalization
        self.adaptive_preference_sampler = adaptive_preference_sampler
        self.normalize_mo_reward_rms = kwargs.get("normalize_mo_reward_rms", False)

        # Multi-media Metric
        self.debug_save_sensor = kwargs.get("debug_save_sensor", False)
        self.debug_save_every_n = kwargs.get("debug_save_every_n", 50)  # 每N步存一张
        self.debug_sensor_dir = Path(kwargs.get("debug_sensor_dir", "debug_sensor_frames"))
        self.debug_sensor_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Unity Environment
        self._init_unity_env(seed, **kwargs)
        super().__init__(device=device)
        # Set Observation Space and Action Space 
        self._make_specs()

 
       


    def _init_unity_env(self, seed: Optional[int], **kwargs):
        
        if self.normalize_mo_reward_rms:
            self.mo_norm_gamma = 0.99
            self.mo_norm_eps =  1e-8
            self.mo_returns = np.zeros((self.mo_reward_dim,), dtype=np.float32)
            self.mo_return_rms = RunningMeanStd(shape=(self.mo_reward_dim,))

        # Render Configuration
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(
            width=kwargs.get("width", 800),
            height=kwargs.get("height", 800),
            time_scale=kwargs.get("time_scale", 10.0)
        )
        # Create Environment
        # for i in range(max_retries):
        #     try:
        #         current_worker_id = self.worker_id + i 
        #         # print(f"🔄 Trying to connect to Unity with worker_id={current_worker_id}")
        #         self.env = UnityEnvironment(
        #             file_name=self.env_path,
        #             seed=seed or 1,
        #             side_channels=[channel],
        #             worker_id=current_worker_id,
        #             no_graphics=not kwargs.get("render", True)
        #         )
        #         self.env_index = current_worker_id
        #         print(f"Connected to Unity with worker_id={current_worker_id}, env_index={self.env_index}")
        #         break
        #     except Exception as e:
        #         current_worker_id += 1
        #         if i == max_retries - 1:
        #             raise RuntimeError(
        #                 f"Failed to connect to Unity after {max_retries} attempts. ")
        #         continue
                
        for i in range(5):
            try:
                current_worker_id = self.worker_id 
                print(
                    f"🔄 Trying to connect to Unity with worker_id={current_worker_id}"
                )
                self.env = UnityEnvironment(
                    file_name=self.env_path,
                    seed=seed or 1,
                    side_channels=[channel],
                    worker_id=current_worker_id,
                    no_graphics=not kwargs.get("render", True)
                )
                self.env_index = current_worker_id
                print(
                    "🟢  Connected to Unity with "
                    f"worker_id={current_worker_id}, env_index={self.env_index}"
                )
                break

            except Exception as e:
                if i == 4:
                    raise RuntimeError(
                        "🔴 Failed to connect to Unity with "
                        f"worker_id={current_worker_id}: {e}"
                    ) from e
                else:
                    # current_worker_id += 1
                    continue
                    
        # Reset Environment and Get Specs
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        # Environment and Agent Info
        self.n_agents = len(decision_steps)
        # Vector Info 45
        self.obs_dim = self.spec.observation_specs[1].shape[0] - self.manual_metrics_size 
        # Image Info 256
        # self.raw_image_dim = self.spec.observation_specs[0].shape # (64, 64, 3)
        self.image_dim = 16 * 16
        self.state_dim = self.obs_dim + self.image_dim # 45 + 256 = 301

        if self.spec.action_spec.is_continuous():
            self.action_dim = self.spec.action_spec.continuous_size 
        else:
            self.action_dim = len(self.spec.action_spec.discrete_branches)
            
        print("----------------Unity Environment Info:----------------")
        print(f"  Agents: {self.n_agents} Continuous Actions: {self.continuous_actions} ")
        print(f"  State Dim: {self.state_dim }") 
        print(f"  Obs Dim: {self.obs_dim} ") 
        print(f"  Image Dim: {self.image_dim} ") 
        print(f"  Preference Dim: {self.mo_reward_dim}")
        print(f"  Action Dim: {self.action_dim}")
        print(f"  Manual Metrics Size: {getattr(self, 'manual_metrics_size', 'N/A')}")
        print("----------------------------------------------------------")

    # TODO: 需要修改SAMPLE PREFERENCE WITH SAMPLER的逻辑
    # def sample_preference_with_sampler(
    #     self,
    #     condition_on_preference: bool,
    #     fixed_preference: np.ndarray | None,
    #     env_type: str,
    #     eval_u: float | None,
    #     adaptive_preference_sampler: SharedPreferenceState,
    # ) -> np.ndarray:
    #     if fixed_preference is not None:
    #         return np.asarray(fixed_preference, dtype=np.float32)
    #     if not condition_on_preference:
    #         return np.array([1.0, 0.0], dtype=np.float32)

    #     # if env_type == "eval" and eval_u is not None:
    #     #     u = float(eval_u)
    #     elif adaptive_preference_sampler is not None:
    #         u = adaptive_preference_sampler.sample_u_train()
    #     else:
    #         u = float(np.random.uniform(0.0, 1.0))

    #     return np.array([u, 1.0 - u], dtype=np.float32)
    def _sample_truncated_normal_preference(self):
        mean_value = 1
        std_value = 0.1
        low_value = 0.0
        high_value = 1.0
        while True:
            w = np.random.normal(mean_value, std_value)
            if low_value <= w <= high_value:
                return np.array([float(w), 1.0 - float(w)], dtype=np.float32)

    def u_sample_preference(self) -> np.ndarray:
        if self.fixed_preference is not None:
            return np.asarray(self.fixed_preference, dtype=np.float32)

        # uniformly sample preference for evaluation
        if self.condition_on_preference and self.env_type == "eval":
            n = self.max_episodes
            k = self.env_index % n         
            u = k / (n - 1) if n > 1 else 0.0
            # print(f"[debug] [{self.env_index}:{self.max_episodes}] env_type: {self.env_type} preference: {u}")
            return np.array([u, 1.0 - u], dtype=np.float32)
            
           
        elif self.condition_on_preference and self.env_type == "train":
            u = np.random.uniform(self.u_sample_start, 1.0)
            # print(f"[debug] [{self.env_index}:{self.max_episodes}] env_type: {self.env_type} start from {self.u_sample_start} sampled preference: {u}")
            return np.array([u, 1.0 - u], dtype=np.float32)
        else:
            raise ValueError(f"Invalid preference sampling method: {self.env_type}")

    def _make_specs(self):
        # Observation Space Specification

        # Grid Version
        observation_spec = Composite({
            "agents": Composite({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(self.n_agents, self.state_dim,),
                    dtype=torch.float32,
                ),
            }, shape=(self.n_agents,))
        })
        if self.condition_on_preference:
            observation_spec["agents"].set(
                "preference",
                UnboundedContinuousTensorSpec(
                    shape=(self.n_agents, self.mo_reward_dim), dtype=torch.float32,
                ),
            )
        observation_spec["reward_vector"]  = UnboundedContinuousTensorSpec(
                shape=(self.n_agents, self.mo_reward_dim),
                dtype=torch.float32,
            )

        observation_spec["manual_metrics"] = UnboundedContinuousTensorSpec(
            shape=(1, self.manual_metrics_size),
            dtype=torch.float32,
        )

        # Image Version
        # observation_spec = Composite({
        #     "agents": Composite({
        #         "observation": UnboundedContinuousTensorSpec(
        #             shape=(self.n_agents, self.obs_dim,),
        #             dtype=torch.float32,
        #         ),
        #         "image": UnboundedContinuousTensorSpec(
        #             shape=(self.n_agents, 64, 64, 3),
        #             dtype=torch.uint8,
        #         ),
        #     }, shape=(self.n_agents,))
        # })
        
        # Action Space Specification
        low = torch.full((self.n_agents, self.action_dim), -1.0, dtype=torch.float32)
        high = torch.full((self.n_agents, self.action_dim), 1.0, dtype=torch.float32)
        action_spec = Composite({
            "agents": Composite({
                "action": BoundedTensorSpec(
                    low=low,
                    high=high,
                    shape=(self.n_agents, self.action_dim,),
                    dtype=torch.float32,
                )
            }, shape=(self.n_agents,) )
        })
        # test_spec = action_spec["agents", "action"]
    
        reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuousTensorSpec(
                    shape=(self.n_agents, 1,),
                    dtype=torch.float32,
                ),
            }, shape=(self.n_agents,))
        })

        done_spec = Composite({
                "done": DiscreteTensorSpec(
                    n=2,
                    shape=(1,),
                    dtype=torch.bool,
                )
            }, shape=(1,))
        done_spec["terminated"] = DiscreteTensorSpec(
            n=2,
            shape=(1,),
            dtype=torch.bool,
        )
        done_spec["truncated"] = DiscreteTensorSpec(
            n=2,
            shape=(1,),
            dtype=torch.bool,
        )
     
        self.full_observation_spec = observation_spec
        self.full_action_spec = action_spec
        self.full_reward_spec = reward_spec
        self.full_done_spec = done_spec

    def _reset(self, tensordict: TensorDict = None, **kwargs) -> TensorDict:
        """重置环境"""
        self.env.reset()
        self._step_count = 0
        self.agents_finished = np.zeros(self.n_agents, dtype=bool)
        # Initialize Rms Reward Normalization
        if self.normalize_mo_reward_rms:
            self.mo_returns.fill(0.0)

        # Sample Preference
        if self.adaptive_preference_sampler is not None:
            self.w1 = self.adaptive_preference_sampler.sample()
            self.current_w = np.array([self.w1, 1.0 - self.w1], dtype=np.float32)
        else:
            # sample according to the type eval and train
            self.current_w = self.u_sample_preference() 
            # self.current_w = self._sample_truncated_normal_preference()

        pref = np.broadcast_to(self.current_w, (self.n_agents, self.mo_reward_dim)).copy()
        
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        # Get Observation
        obs = decision_steps.obs[1][:, :self.obs_dim] # shape: (n_agents, obs_dim)
        image = decision_steps.obs[0] # shape: (n_agents, image_dim[0], image_dim[1], image_dim[2]) 
        if self.debug_save_sensor:
            self._save_obs0_as_gray(image, tag="reset")
        flattened_image = self.downsample_image(image, out_h=int(np.sqrt(self.image_dim)), out_w=int(np.sqrt(self.image_dim)))
        state = np.concatenate([flattened_image, obs], axis=1) # shape: (n_agents, 256 + 45 = 301)

        dones = torch.zeros((self.n_agents,1), dtype=torch.bool, device=self.device)
        global_done = torch.zeros(1, dtype=torch.bool, device=self.device)
        global_terminated = torch.zeros(1, dtype=torch.bool, device=self.device)
        global_truncated = torch.zeros(1, dtype=torch.bool, device=self.device)
        manual_metrics = torch.zeros(
            (1, self.manual_metrics_size), dtype=torch.float32, device=self.device
        )

        agent_dict = {
            "agents": {
                "observation": torch.tensor(state, dtype=torch.float32, device=self.device),
                "reward": torch.zeros((self.n_agents, 1), dtype=torch.float32, device=self.device),
                "done_list": dones, 
                "done": dones     
            },
            "reward_vector": torch.zeros((self.n_agents, self.mo_reward_dim), dtype=torch.float32, device=self.device,),
            "manual_metrics": manual_metrics, 
            "done": global_done,
            "terminated": global_terminated,
            "truncated": global_truncated
        }
        if self.condition_on_preference:
            agent_dict["agents"]["preference"] = torch.tensor(pref, dtype=torch.float32, device=self.device)

        return TensorDict(agent_dict, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """执行一步动作"""
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        actions = tensordict["agents"]["action"]
        actions_np = actions.cpu().numpy()

        if len(decision_steps) > 0:
            current_actions = np.zeros((len(decision_steps), self.action_dim), dtype=np.float32)
            for i, agent_id in enumerate(decision_steps.agent_id):
                # print(f"agent_id: {agent_id}")
                current_actions[i] = actions_np[i]

            action_tuple = ActionTuple()
            if self.continuous_actions:
                action_tuple.add_continuous(current_actions)
            else:
                action_tuple.add_discrete(current_actions.astype(np.int32))

            self.env.set_actions(self.behavior_name, action_tuple)
  
        self.env.step()
        
        # 获取结果
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if self.debug_save_sensor and (self._step_count % self.debug_save_every_n == 0):
            if len(decision_steps.agent_id) > 0:
                self._save_obs0_as_gray(decision_steps.obs[0], tag="decision")
            elif len(terminal_steps.agent_id) > 0:
                self._save_obs0_as_gray(terminal_steps.obs[0], tag="terminal")

        if len(decision_steps.agent_id) > 0:
            dec_image = self.downsample_image(decision_steps.obs[0], out_h=int(np.sqrt(self.image_dim)), out_w=int(np.sqrt(self.image_dim)))
        if len(terminal_steps.agent_id) > 0:
            ter_image = self.downsample_image(terminal_steps.obs[0], out_h=int(np.sqrt(self.image_dim)), out_w=int(np.sqrt(self.image_dim)))
   
        unity_total_dim = self.state_dim + self.manual_metrics_size # 301 + 6 = 307
        full_obs = np.zeros((self.n_agents, unity_total_dim), dtype=np.float32)

        # Objectives: exploration increase and path penalty.
        vector_rewards = np.zeros(
            (self.n_agents, self.mo_reward_dim), dtype=np.float32
        )
           
        # Combine Decision Steps and Terminal Steps (image + obs)
        for agent_id in decision_steps.agent_id:
            # if agent_id < self.n_agents:
            idx = decision_steps.agent_id_to_index[agent_id]
            full_obs[idx] = np.concatenate([dec_image[idx], decision_steps.obs[1][idx]], axis=0) 
            # rewards[agent_id] = decision_steps.reward[idx]
            vector_rewards[idx][0] = decision_steps.reward[idx] * 0.1

        for agent_id in terminal_steps.agent_id:
            # if agent_id < self.n_agents:
            idx = terminal_steps.agent_id_to_index[agent_id]
            full_obs[idx] = np.concatenate([ter_image[idx], terminal_steps.obs[1][idx]], axis=0)
            # rewards[agent_id] = terminal_steps.reward[idx]
            vector_rewards[idx][0] = terminal_steps.reward[idx] * 0.1
            
        vector_rewards[:, 0] = vector_rewards[:, 0].max()
        # pathpenalty = -full_obs[:, -self.manual_metrics_size].mean()  ## SHARED
        pathpenalty = -full_obs[:, -self.manual_metrics_size]  ##HYbrid
        vector_rewards[:, 1] = pathpenalty
        
      
        metrics_raw =  full_obs[:, -self.manual_metrics_size:] # shape: (n_agents, 6)
        manual_metrics = np.mean(metrics_raw, axis=0) # Average over agents
        # last manual metrics index represent the exploration ratio
        manual_metrics[-1] = metrics_raw[:, -1].max()
        # print(f"full_obs: \n {full_obs[:, -1]}")
   
        is_success = full_obs[:, -1].max() >= 95.0
  
        # self.agents_finishesd = self.agents_finished | is_success
        self._step_count += 1

        all_success = is_success
        is_timeout = self._step_count >= self.max_steps
        # agents_done_np = self.agents_finished | is_timeout

        global_done_bool = bool(all_success or is_timeout) # task done (success or timeout)
        agents_done_np = np.full(self.n_agents, global_done_bool)
        global_terminated_bool = bool(all_success)      # task success
        global_truncated_bool = bool(is_timeout and not all_success)  # timeout
  
        # Multi-ojectives Reward Normalization
        if self.normalize_mo_reward_rms:
            vector_rewards_used = self._normalize_mo_reward_rms(
                vector_rewards=vector_rewards,
                done=bool(global_done_bool),)
        else:
            vector_rewards_used = vector_rewards

        w = self.current_w   
        rewards = np.sum(vector_rewards_used * w[None, :], axis=1)
        pref = np.broadcast_to(self.current_w, (self.n_agents, self.mo_reward_dim)).copy()
        # if self._step_count % 50 == 0:
            # print(f"agent 0 state: {full_obs[0, :self.state_dim]} \n")
            # print(f"Manual Metrics Averaged: {manual_metrics} \n")
            # print(f"raw rewards: {vector_rewards} \n")
            # print(f"normalized rewards: {vector_rewards_used} \n")
            # print(f"{self._step_count} scaled rewards: {rewards} \n")
        # print(f"Step {self._step_count}: Rewards: {rewards}\n")
            # print(f"raw_rewards:  {vector_rewards_used} \n preference:  {w}\n rewards:  {rewards} \n exploration:  {full_obs[:, -1]} \n manual_metrics:  {manual_metrics}")   
            #  print(f"Step {self._step_count}: Rewards: {rewards}, Is Success: {agents_done_np}, all success {global_done_bool} Timeout: {is_timeout} \n")
        # Observation t+1
        next_obs = full_obs[:, :self.state_dim] # shape: (n_agents, 301)

     
        
        agent_dict = {
            "agents": {
                "observation": torch.tensor(next_obs, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device).view(self.n_agents, 1),
                "done_list": torch.tensor(agents_done_np, dtype=torch.bool, device=self.device).view(self.n_agents, 1),
               },
            "reward_vector": torch.tensor(vector_rewards_used, dtype=torch.float32, device=self.device,).view(self.n_agents, self.mo_reward_dim),
            "manual_metrics": torch.tensor(
                manual_metrics, dtype=torch.float32, device=self.device
            ).view(1, -1),
            "done": torch.tensor(
                global_done_bool, dtype=torch.bool, device=self.device
            ).view(1),
            "terminated": torch.tensor(
                global_terminated_bool, dtype=torch.bool, device=self.device
            ).view(1),
            "truncated": torch.tensor(
                global_truncated_bool, dtype=torch.bool, device=self.device
            ).view(1),
        }
        if self.condition_on_preference:
            agent_dict["agents"]["preference"] = torch.tensor(pref, dtype=torch.float32, device=self.device)
            
        return TensorDict(agent_dict, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: Optional[int]):
        """设置随机种子"""
        self.seed = seed

    def close(self, **kwargs): 
        """关闭环境 (支持 **kwargs 以兼容 TorchRL)"""
        if hasattr(self, 'env'):
            try:
             
                if hasattr(self.env, '_close'):
                    atexit.unregister(self.env._close)
                self.env.close()
                # print("Closing Unity Environment...成功关闭")
            except Exception:
                pass

    def render(self, mode=None, **kwargs):
        import mss
        with mss.mss() as sct:
            # 获取主屏幕
            monitor = sct.monitors[1]
            # 截图
            img = np.array(sct.grab(monitor))
            # 转换格式: BGRA -> RGB
            img = img[:, :, :3][:, :, ::-1]
            
            # 可选：调整大小以减小视频体积
            # import cv2
            # img = cv2.resize(img, (640, 480))
            return img
        
    @property
    def agents(self) -> List[str]:
        """返回智能体名称列表"""
        return [f"agent_{i}" for i in range(self.n_agents)]

    def __del__(self):
        """析构函数"""
        self.close()


    def _save_obs0_as_gray(self, obs0_np: np.ndarray, tag: str = "step") -> Optional[str]:
        """
        obs0_np: decision_steps.obs[0] 或 terminal_steps.obs[0] 的 numpy
        目标: 保存成黑白二维图，便于快速验证读取是否正常
        """
        if isinstance(obs0_np, torch.Tensor):
            obs0_np = obs0_np.detach().cpu().numpy()
        
        if obs0_np is None or obs0_np.size == 0:
            print(f"[debug_sensor] skip {tag}: empty obs0")
            return None

        # 取第一个agent
        x = obs0_np[0]
        print(
            f"[debug_sensor] {tag} step={self._step_count} "
            f"obs0 shape={obs0_np.shape} agent0 shape={x.shape} "
            f"min={float(np.min(x)):.4f} max={float(np.max(x)):.4f}"
        )

        # 常见情况1: render texture 展平为一维向量
        if x.ndim == 1:
            # 尝试推断为方图（例如 128*128）
            side = int(np.sqrt(x.shape[0]))
            if side * side == x.shape[0]:
                img = x.reshape(side, side)
            else:
                # 兜底：当成一行，至少能看数据是否变化
                img = x.reshape(1, -1)

        # 常见情况2: 已经是 HxW 或 HxWxC
        elif x.ndim == 2:
            img = x
        elif x.ndim == 3:
            # 转灰度
            img = x.mean(axis=-1)
        else:
            print(f"[debug_sensor] skip {tag}: unsupported ndim={x.ndim}")
            return None

        # 归一化到 0~255
        img = img.astype(np.float32)
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        img_u8 = (img * 255).astype(np.uint8)

        out = self.debug_sensor_dir / f"{tag}_{self._step_count:06d}.png"
        imageio.imwrite(out, img_u8)
        print(f"[debug_sensor] saved -> {out.resolve()}")
        return str(out.resolve())

    def downsample_image(self, image, out_h=16, out_w=16):
        # Unity 观测在 CPU 上，这里用 cpu 做池化再转 numpy，避免无意义的 GPU 往返
        x = torch.as_tensor(image, dtype=torch.float32, device="cpu")
        single = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            single = True
        x = x.mean(dim=-1, keepdim=True)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x, output_size=(out_h, out_w))
        # ML-Agents visual observations are normally already float32 in
        # [0, 1]. Keep compatibility with raw uint8-like arrays without
        # shrinking normalized observations by another factor of 255.
        if x.numel() and x.max() > 1.0:
            x = x / 255.0
        x = x.flatten(start_dim=1)
        arr = x.numpy().astype(np.float32, copy=False)
        if single:
            arr = arr.squeeze(0)
        return arr

    def _normalize_mo_reward_rms(
    self,
    vector_rewards: np.ndarray,   # shape: (n_agents, 3)
    done: bool,                   # global done for this env step
    ) -> np.ndarray:
        """
        1) Update discounted returns: R_t = gamma * R_{t-1} * (1-done) + r_t
        2) Update running stats with returns
        3) Normalize immediate rewards using return variance
        """
        # Aggregate per-objective immediate reward across agents for return tracking
        r_obj = vector_rewards.mean(axis=0).astype(np.float32)  # (3,)
        self.mo_returns = self.mo_returns * self.mo_norm_gamma * (1 - done) + r_obj
      
        # Update return RMS (batch-like shape)
        self.mo_return_rms.update(self.mo_returns[None, :])
       
        # Normalize immediate reward with return variance
        scale = np.sqrt(self.mo_return_rms.var + self.mo_norm_eps).astype(np.float32)  # (3,)
        vector_rewards_norm = vector_rewards / scale[None, :]  # keep shape (n_agents, 3)
        # print(f"mo_returns: {self.mo_returns}, self.mo_return_rms: {self.mo_return_rms.var}, scale: {scale}")
        # print(f"vector_rewards: {vector_rewards}, vector_rewards_norm: {vector_rewards_norm}")
        return vector_rewards_norm.astype(np.float32)
