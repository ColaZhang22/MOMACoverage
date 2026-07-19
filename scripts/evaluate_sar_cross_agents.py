"""Evaluate a SAR checkpoint on a different SAR agent-count task.

Example:
    python scripts/evaluate_sar_cross_agents.py ^
        outputs/.../your_exp/checkpoints/checkpoint_200000.pt ^
        --tasks moroom128cpu2a moroom128cpu3a ^
        --episodes 32
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Older mlagents_envs releases still reference np.bool, which NumPy removed in
# 1.24. Keep this local compatibility shim in the evaluation entry point.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarl.environments import SARTask
from benchmarl.evaluations.mo_metrics import compute_explore_path_mo_metrics
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback


METRIC_NAMES = [
    "DeltaPathLengthObs",
    "StepCount",
    "PathLength",
    "CollisionWallCount",
    "CollisionAgentCount",
    "ExplorationRatio",
]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if rank_a.std() == 0.0 or rank_b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = values.min()
    high = values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _arc_length_parameter(objective_1: np.ndarray, objective_2: np.ndarray) -> np.ndarray:
    x = _minmax(objective_1)
    y = _minmax(objective_2)
    order = np.lexsort((-y, x))
    points = np.column_stack([x[order], y[order]])

    segments = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    if cumulative[-1] > 0.0:
        cumulative = cumulative / cumulative[-1]

    parameter = np.empty(len(x), dtype=float)
    parameter[order] = cumulative
    return parameter


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return value


class RolloutCaptureCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.rollouts = []

    def on_evaluation_end(self, rollouts):
        self.rollouts = list(rollouts)


def _load_saved_config(checkpoint_file: Path) -> dict[str, Any]:
    config_file = checkpoint_file.parent.parent / "config.pkl"
    if not config_file.exists():
        raise FileNotFoundError(f"Cannot find config.pkl next to checkpoint: {config_file}")

    with config_file.open("rb") as f:
        task = pickle.load(f)
        task_config = pickle.load(f)
        algorithm_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks = pickle.load(f)

    return {
        "task": task,
        "task_config": task_config,
        "algorithm_config": algorithm_config,
        "model_config": model_config,
        "seed": seed,
        "experiment_config": experiment_config,
        "critic_model_config": critic_model_config,
        "callbacks": callbacks,
    }


def _sar_task_from_name(task_name: str):
    enum_name = task_name.upper()
    if hasattr(SARTask, enum_name):
        return getattr(SARTask, enum_name).get_from_yaml()

    yaml_path = REPO_ROOT / "benchmarl" / "conf" / "task" / "sar" / f"{task_name}.yaml"
    if yaml_path.exists():
        return SARTask.MOROOM128CPU.get_from_yaml(str(yaml_path))

    if not hasattr(SARTask, enum_name):
        choices = ", ".join(t.name.lower() for t in SARTask)
        raise ValueError(
            f"Unknown SAR task '{task_name}'. Available SAR tasks: {choices}. "
            f"Also looked for yaml: {yaml_path}"
        )


def _last_done_index(rollout) -> int:
    done = rollout["next", "done"].squeeze(-1)
    done_steps = done.nonzero(as_tuple=True)[0]
    if done_steps.numel() > 0:
        return int(done_steps[0].item())
    return int(done.shape[0] - 1)


def _summarize_rollouts(
    rollouts,
    minmax_points: list[list[float]],
    *,
    run_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    success_count = 0

    for episode, rollout in enumerate(rollouts):
        end_idx = _last_done_index(rollout)
        terminated = bool(rollout["next", "terminated"].squeeze(-1)[end_idx].item())
        truncated = bool(rollout["next", "truncated"].squeeze(-1)[end_idx].item())
        success = terminated and not truncated
        success_count += int(success)

        manual_metrics = rollout["next", "manual_metrics"][end_idx].reshape(-1)
        row = {
            "episode": episode,
            "id": run_name,
            "name": run_name,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
        }

        for i, name in enumerate(METRIC_NAMES):
            if i < manual_metrics.shape[0]:
                row[name] = float(manual_metrics[i].item())

        if "ExplorationRatio" in row:
            row["exploration_ratio"] = row["ExplorationRatio"]
        if "PathLength" in row:
            row["neg_path_length"] = -row["PathLength"]

        try:
            row["preference_u"] = float(rollout["agents", "preference"][0, 0, 0].item())
        except KeyError:
            pass

        rows.append(row)

    mo = compute_explore_path_mo_metrics(rollouts, minmax_points)
    averages = {
        name: float(np.mean([r[name] for r in rows if name in r]))
        for name in METRIC_NAMES
        if any(name in r for r in rows)
    }
    pas = {}
    if all(
        all(column in row for column in ("preference_u", "exploration_ratio", "neg_path_length"))
        for row in rows
    ):
        preference = np.asarray([row["preference_u"] for row in rows], dtype=float)
        objective_1 = np.asarray([row["exploration_ratio"] for row in rows], dtype=float)
        objective_2 = np.asarray([row["neg_path_length"] for row in rows], dtype=float)
        arc_parameter = _arc_length_parameter(objective_1, objective_2)
        pas = {
            "PAS_x": _spearman(preference, objective_1),
            "PAS_arc": _spearman(preference, arc_parameter),
            "u_min": float(preference.min()) if len(preference) else float("nan"),
            "u_max": float(preference.max()) if len(preference) else float("nan"),
        }
        for row, arc_value in zip(rows, arc_parameter):
            row["pas_arc_parameter"] = float(arc_value)

    summary = {
        "n_episodes": len(rollouts),
        "win_rate": success_count / len(rollouts) if rollouts else 0.0,
        "mo_hypervolume": float(mo["hv"]),
        "mo_cardinality": int(mo["cardinality"]),
        "mo_sparsity": float(mo["sparsity"]),
        **pas,
        "mo_points": mo["points"].tolist(),
        "averages": averages,
    }
    return summary, rows


def _write_results(output_dir: Path, task_name: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"{task_name}_summary.json"
    rows_file = output_dir / f"{task_name}_episodes.csv"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(_safe_json_value(summary), f, indent=2)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with rows_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary: {summary_file}")
    print(f"Saved per-episode metrics: {rows_file}")


def _copy_matching_state_dict(source: dict[str, Any], target_module, *, source_prefix: str = "") -> int:
    target_state = target_module.state_dict()
    merged_state = dict(target_state)
    loaded = 0

    for source_key, source_value in source.items():
        if source_prefix:
            if not source_key.startswith(source_prefix):
                continue
            target_key = source_key[len(source_prefix):]
        else:
            target_key = source_key
        if target_key.startswith("__"):
            continue
        if target_key not in target_state:
            continue
        target_value = target_state[target_key]
        if not hasattr(source_value, "shape") or not hasattr(target_value, "shape"):
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            continue
        merged_state[target_key] = source_value.to(target_value.device)
        loaded += 1

    if loaded:
        target_module.load_state_dict(merged_state, strict=False)
    return loaded


def _load_actor_only(experiment: Experiment, checkpoint_file: Path) -> None:
    checkpoint = torch.load(checkpoint_file, map_location=experiment.config.train_device)
    for group in experiment.group_map.keys():
        loss_key = f"loss_{group}"
        if loss_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain {loss_key}")

        saved_loss_state = checkpoint[loss_key]
        current_loss_state = experiment.losses[group].state_dict()
        merged_loss_state = dict(current_loss_state)
        loaded_loss_keys = 0

        if any(key.startswith("actor_network_params.") for key in saved_loss_state):
            actor_prefixes = ("actor_network_params.",)
        elif any(key.startswith("actor.") for key in saved_loss_state):
            actor_prefixes = ("actor.", "planner.")
        else:
            actor_prefixes = ("actor.",)

        for key, value in saved_loss_state.items():
            if not key.startswith(actor_prefixes):
                continue
            if key not in current_loss_state:
                continue
            current_value = current_loss_state[key]
            if not hasattr(value, "shape") or not hasattr(current_value, "shape"):
                continue
            if tuple(value.shape) != tuple(current_value.shape):
                continue
            merged_loss_state[key] = value.to(current_value.device)
            loaded_loss_keys += 1

        if loaded_loss_keys == 0:
            raise RuntimeError(
                f"No actor keys from {loss_key} matched the current {group} actor. "
                "The actor architecture is not compatible."
            )
        experiment.losses[group].load_state_dict(merged_loss_state, strict=False)

        loaded_policy_keys = _copy_matching_state_dict(
            saved_loss_state,
            experiment.group_policies[group],
            source_prefix="actor_network_params.",
        ) if actor_prefixes == ("actor_network_params.",) else 0
        loaded_full_policy_keys = (
            _copy_matching_state_dict(
                saved_loss_state,
                experiment.policy,
                source_prefix="actor_network_params.",
            )
            if actor_prefixes == ("actor_network_params.",)
            else 0
        )
        print(
            f"Loaded actor-only weights for group '{group}': "
            f"{loaded_loss_keys} loss keys, "
            f"{loaded_policy_keys} group-policy keys, "
            f"{loaded_full_policy_keys} full-policy keys."
        )


def _safe_close_experiment(experiment: Experiment | None) -> None:
    if experiment is None:
        return
    try:
        experiment.close()
    except Exception as exc:
        print(f"Warning: experiment.close() failed: {exc}")
        try:
            experiment.logger.finish()
        except Exception:
            pass


def evaluate_on_task(
    checkpoint_file: Path,
    saved: dict[str, Any],
    task_name: str,
    output_root: Path,
    episodes: int | None,
    parallel: bool,
    keep_saved_callbacks: bool,
    use_original_loggers: bool,
    time_scale: float | None,
    render: bool,
    max_steps: int | None,
    actor_only: bool,
) -> dict[str, Any]:
    target_task = _sar_task_from_name(task_name)
    if time_scale is not None:
        target_task.config["time_scale"] = time_scale
    if max_steps is not None:
        target_task.config["max_steps"] = max_steps
    target_task.config["render"] = render
    experiment_config = copy.deepcopy(saved["experiment_config"])
    experiment_config.restore_file = None if actor_only else str(checkpoint_file)
    experiment_config.save_folder = str(output_root / task_name)
    Path(experiment_config.save_folder).mkdir(parents=True, exist_ok=True)
    experiment_config.render = render
    experiment_config.parallel_evaluation = parallel
    experiment_config.evaluation = True
    experiment_config.checkpoint_interval = 0
    experiment_config.checkpoint_at_end = False
    experiment_config.adaptive_preference_sampler = False
    experiment_config.on_policy_n_envs_per_worker = 1
    experiment_config.off_policy_n_envs_per_worker = 1
    experiment_config.parallel_collection = False
    if episodes is not None:
        experiment_config.evaluation_episodes = episodes
    if not use_original_loggers:
        experiment_config.loggers = ["csv"]
        experiment_config.create_json = True
        experiment_config.wandb_extra_kwargs = {}

    capture = RolloutCaptureCallback()
    callbacks = list(saved["callbacks"]) if keep_saved_callbacks else []
    callbacks.append(capture)

    print(f"\n=== Evaluating {checkpoint_file.name} on sar/{task_name} ===")
    print(
        f"Agents: {target_task.config.get('n_agents')}  "
        f"Episodes: {experiment_config.evaluation_episodes}  "
        f"Time scale: {target_task.config.get('time_scale')}  "
        f"Max steps: {target_task.config.get('max_steps')}  "
        f"Render: {target_task.config.get('render')}"
    )

    experiment = None
    try:
        experiment = Experiment(
            task=target_task,
            algorithm_config=copy.deepcopy(saved["algorithm_config"]),
            model_config=copy.deepcopy(saved["model_config"]),
            seed=saved["seed"],
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=copy.deepcopy(saved["critic_model_config"]),
        )
        if actor_only:
            _load_actor_only(experiment, checkpoint_file)
        experiment.evaluate()

        minmax_points = target_task.config.get("minmax_points", [[0.0, 0.0], [0.0, 0.0]])
        summary, rows = _summarize_rollouts(
            capture.rollouts,
            minmax_points,
            run_name=experiment.name,
        )
        _write_results(output_root, task_name, summary, rows)
    finally:
        _safe_close_experiment(experiment)

    print(
        "Result: "
        f"win_rate={summary['win_rate']:.4f}, "
        f"hv={summary['mo_hypervolume']:.4f}, "
        f"cardinality={summary['mo_cardinality']}, "
        f"sparsity={summary['mo_sparsity']:.4f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 4-agent SAR checkpoint on 2-agent/3-agent SAR Unity tasks."
    )
    parser.add_argument("checkpoint_file", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["moroom128cpu2a", "moroom128cpu3a"],
        help="SAR task yaml names to test, e.g. moroom128cpu2a moroom128cpu3a.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "cross_agent_eval")
    parser.add_argument("--serial", action="store_true", help="Disable parallel evaluation.")
    parser.add_argument("--time-scale", type=float, default=None, help="Override SAR Unity time_scale for evaluation.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override SAR max_steps for smoke tests.")
    parser.add_argument("--render", action="store_true", help="Open the Unity render window during evaluation.")
    parser.add_argument("--actor-only", action="store_true", help="Load only compatible actor weights from the checkpoint.")
    parser.add_argument("--keep-saved-callbacks", action="store_true")
    parser.add_argument("--use-original-loggers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_file = args.checkpoint_file.resolve()
    saved = _load_saved_config(checkpoint_file)
    output_root = args.output_dir.resolve()

    all_results = {}
    for task_name in args.tasks:
        all_results[task_name] = evaluate_on_task(
            checkpoint_file=checkpoint_file,
            saved=saved,
            task_name=task_name,
            output_root=output_root,
            episodes=args.episodes,
            parallel=not args.serial,
            keep_saved_callbacks=args.keep_saved_callbacks,
            use_original_loggers=args.use_original_loggers,
            time_scale=args.time_scale,
            render=args.render,
            max_steps=args.max_steps,
            actor_only=args.actor_only,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    combined_name = (
        f"{args.tasks[0]}_combined_summary.json"
        if len(args.tasks) == 1
        else "combined_summary.json"
    )
    combined_file = output_root / combined_name
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(_safe_json_value(all_results), f, indent=2)
    print(f"\nSaved combined summary: {combined_file}")


if __name__ == "__main__":
    main()
