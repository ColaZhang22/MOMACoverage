"""Manual SAR training entrypoint.

Examples:
    python scripts/train_sar_manual.py --algorithm cmomappo --task moroom128cpu --seed 0
    python scripts/train_sar_manual.py --algorithm pcma --task moroom128cpu --seed 42 --frames 1000000
    python scripts/train_sar_manual.py --algorithm momaac --task moroom128cpu --seed 3407 --no-wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import SARTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.utils import _read_yaml_config


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "benchmarl" / "conf" / "task" / "sar"
EXPERIMENT_DIR = ROOT / "benchmarl" / "conf" / "experiment"


TASK_ENUM = {
    "moroom128cpu": SARTask.MOROOM128CPU,
    "moempty128cpu": SARTask.MOEMPTY128CPU,
    "momaze128cpu": SARTask.MOMAZE128CPU,
    # Use the 4-agent enum as a loader shell; the yaml path below supplies
    # the actual task config, including n_agents.
    "moroom128cpu2a": SARTask.MOROOM128CPU,
    "moroom128cpu3a": SARTask.MOROOM128CPU,
}


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key] = parse_value(value)
    return overrides


def load_cms_experiment_config() -> ExperimentConfig:
    base = _read_yaml_config(str(EXPERIMENT_DIR / "base_experiment.yaml"))
    cms = _read_yaml_config(str(EXPERIMENT_DIR / "sar" / "cms.yaml"))
    base.update(cms)
    base.pop("defaults", None)
    return ExperimentConfig(**base)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_task(task_name: str, env_path: str | None):
    task_name = task_name.lower()
    if task_name not in TASK_ENUM:
        choices = ", ".join(sorted(TASK_ENUM))
        raise ValueError(f"Unknown task '{task_name}'. Choices: {choices}")

    yaml_path = TASK_DIR / f"{task_name}.yaml"
    task = TASK_ENUM[task_name].get_from_yaml(path=str(yaml_path))
    if env_path is not None:
        task.config["env_path"] = env_path
    return task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAR manually without Hydra.")
    parser.add_argument(
        "--algorithm",
        choices=sorted(algorithm_config_registry),
        default="cmomappo",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_ENUM),
        default="moroom128cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--env-path",
        default=None,
        help="Override Unity executable path from the task yaml.",
    )
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=200_000)
    parser.add_argument("--keep-checkpoints-num", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Use one Unity env for collection/evaluation. Slower but easier to debug.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override an ExperimentConfig field, e.g. --override lr=0.0001",
    )
    parser.add_argument(
        "--task-override",
        action="append",
        default=[],
        help="Override a task config field, e.g. --task-override time_scale=1.0",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    task = load_task(args.task, args.env_path)
    task.config.update(parse_overrides(args.task_override))

    algorithm_config = algorithm_config_registry[args.algorithm].get_from_yaml()

    experiment_config = load_cms_experiment_config()
    experiment_config.max_n_frames = args.frames
    experiment_config.max_n_iters = None
    experiment_config.train_device = resolve_device(args.device)
    experiment_config.sampling_device = "cpu"
    experiment_config.buffer_device = "cpu"
    experiment_config.render = args.render
    experiment_config.evaluation = not args.no_eval
    experiment_config.checkpoint_interval = args.checkpoint_interval
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = args.keep_checkpoints_num

    if args.save_folder is not None:
        experiment_config.save_folder = str(Path(args.save_folder).resolve())

    if args.no_wandb:
        experiment_config.loggers = ["csv"]

    if args.serial:
        experiment_config.parallel_collection = False
        experiment_config.parallel_evaluation = False
        experiment_config.on_policy_n_envs_per_worker = 1
        experiment_config.off_policy_n_envs_per_worker = 1

    for key, value in parse_overrides(args.override).items():
        if not hasattr(experiment_config, key):
            raise ValueError(f"ExperimentConfig has no field '{key}'")
        setattr(experiment_config, key, value)

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
