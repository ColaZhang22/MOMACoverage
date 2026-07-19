"""Aggregate cross-agent actor-only evaluation summaries into a LaTeX table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RUNS = {
    "PCMAPPO": ("actor_only_cms_eval", "cmomappo_seed_*"),
    "PCMA": ("pcma_actor_only_cms_eval", "pcma_seed_*"),
    "MOMA-AC": ("momaac_actor_only_cms_eval", "momaac_seed_*"),
}

TASKS = {
    "Room~(3 agents)": "moroom128cpu3a",
    "Room~(2 agents)": "moroom128cpu2a",
}

METRICS = [
    ("HV $\\uparrow$", "mo_hypervolume"),
    ("C $\\uparrow$", "mo_cardinality"),
    ("GEU $\\uparrow$", "GEU"),
    ("PAS $\\uparrow$", "PAS_arc"),
]


def _read_geu(summary_file: Path) -> float | None:
    root = summary_file.parent
    task = summary_file.name.replace("_summary.json", "")
    reward_files = list(
        (root / task).glob("*/**/scalars/eval_agents_reward_episode_reward_mean.csv")
    )
    if not reward_files:
        return None
    reward_file = max(reward_files, key=lambda path: path.stat().st_mtime)
    line = reward_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    return float(line.split(",")[-1])


def _collect(base_dir: Path) -> dict:
    data = {}
    for algorithm, patterns in RUNS.items():
        data[algorithm] = {}
        for task_label, task_name in TASKS.items():
            values = {key: [] for _, key in METRICS}
            summary_files = []
            for pattern in patterns:
                summary_files.extend(base_dir.glob(f"{pattern}/{task_name}_summary.json"))
            for summary_file in sorted(set(summary_files)):
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                for _, key in METRICS:
                    if key == "GEU":
                        value = _read_geu(summary_file)
                    else:
                        value = summary.get(key)
                    if value is not None:
                        values[key].append(float(value))
            data[algorithm][task_label] = values
    return data


def _format(values: list[float], *, integer: bool = False) -> str:
    if not values:
        return "$-$"
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if integer:
        return f"${mean:.2f} \\pm {std:.2f}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def _latex(data: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-agent transfer performance of 4-agent MOMARL actors in MOMACoverage Room.}",
        r"\label{tab:room_cross_agent_transfer}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{c|c|ccc}",
        r"\toprule",
        r"Task & Metrics & PCMAPPO & PCMA & MOMA-AC \\",
        r"\midrule",
    ]

    task_items = list(TASKS.items())
    for task_index, (task_label, _) in enumerate(task_items):
        for metric_index, (metric_label, key) in enumerate(METRICS):
            prefix = (
                rf"\multirow{{4}}{{*}}{{{task_label}}}"
                if metric_index == 0
                else ""
            )
            row_values = [
                _format(data[algorithm][task_label][key], integer=(key == "mo_cardinality"))
                for algorithm in RUNS
            ]
            lines.append(
                f"{prefix} & {metric_label} & "
                + " & ".join(row_values)
                + r" \\"
            )
            if metric_index != len(METRICS) - 1:
                lines.append("")
        if task_index != len(task_items) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/cross_agent_eval"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = _collect(args.base_dir)
    latex = _latex(data)
    print(latex)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
