"""Run MOMA-TD3 on the existing preference-conditioned SAR environment."""

from benchmarl.algorithms.moma_ac import MomaACConfig
from benchmarl.environments import SARTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.gamma = 0.99
    experiment_config.lr = 3e-4
    experiment_config.soft_target_update = True
    experiment_config.polyak_tau = 0.005
    experiment_config.off_policy_init_random_frames = 25_000

    # This task already emits:
    #   ("agents", "preference") and top-level "reward_vector".
    task = SARTask.MOROOM128CPU.get_from_yaml()
    task.config["condition_on_preference"] = True
    task.config["fixed_preference"] = None

    algorithm_config = MomaACConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
