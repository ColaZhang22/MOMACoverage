![MOMACoverage title](./assets/icon.png)

## Overview 🧐

MOMACoverage is a MOMARL benchmark for continuous cooperative multi-agent exploration with two intrinsic conflicting objectives, map coverage and path length. Agents in MOMACoverage need to cooperatively maximize environment coverage while minimizing the mean path length. 

MOMACoverage is build upon on [BenchMARL](https://github.com/facebookresearch/BenchMARL) 1.5.1 with a Unity [ML-Agents toolkit](https://github.com/unity-technologies/ml-agents).

Pareto points in MOMACoverage are defined as:

```text
F = [ExplorationRatio, -PathLength]
```

Both dimensions are treated as maximization objectives.


1️⃣ Installation

```cmd
# Clone the repository
git clone https://github.com/ColaZhang22/MOMACoverage.git
cd MOMACoverage
```

```cmd
# Create conda environment
conda create -n MOMACoverage python=3.10 -y
conda activate MOMACoverage

pip install -r requirements.txt
```


2️⃣ Downlaod Maps and Add Path to Yaml file

Unity builds are not included in this repository, please download `maps` from the following link: [Google Drive](https://drive.google.com/drive/folders/1SGUg8nzm18HtZm1xMEH6AjKDcpDwedAp?usp=drive_link).

Open `benchmarl/conf/task/sar/xxx.yaml` and update the env_path field:
```yaml
env_path:  fill the downloaded env path, such as "E:\\PythonCode\\MOCMSENV\\MOROOM128CPU\\UnityEnvironment"
```

### Current Task in MAMOCoverage

| Experiment         | Hydra task          | Agents | Normalization anchors    |
| ------------------ | ------------------- | ------ | ------------------------ |
| Main Room task     | `sar/moroom128cpu`  | 4      | `[[105, 0], [0, -1100]]` |
| Main Maze task     | `sar/momaze128cpu`  | 4      | `[[105, 0], [0, -1000]]` |
| Main Empty task    | `sar/moempty128cpu` | 4      | `[[105, 0], [0, -1100]]` |
| Room transfer      | `moroom128cpu3a`    | 3      | `[[105, 0], [0, -1100]]` |
| Room transfer      | `moroom128cpu2a`    | 2      | `[[105, 0], [0, -1100]]` |
| Map generalization | `morandom128cpu`    | 4      | `[[105, 0], [0, -1100]]` |
| Map generalization | `moden128cpu`       | 4      | `[[105, 0], [0, -1100]]` |
| Map generalization | `moden3128cpu`      | 4      | `[[105, 0], [0, -1100]]` |

Update the appropriate file under `benchmarl/conf/task/sar/`, or override the path temporarily through the manual training entry point:

```powershell
python scripts/train_sar_manual.py `
  --algorithm cmomappo `
  --task moroom128cpu `
  --env-path "D:\MOMACoverage\MOROOM128CPU\UnityEnvironment" `
  --seed 0
```

Headless training is controlled by `render: false` in the task configuration. The current `time_scale` is `1.0`.

## 5. Training


## Tasks and algorithms
MOMAcoverage supports both MARL and MOMARL algorithm, and has intergrated several classical MOMARL algorithm into framework. 

| Name used in the README/paper | `algorithm=` | Type       | Description                                 |
| ----------------------------- | ------------ | ---------- | ------------------------------------------- |
| PCMAPPO                       | `cmomappo`   | On-policy  | Preference-conditioned MAPPO                |
| PCMA                          | `pcma`       | On-policy  | Global/local preference coordination policy |
| MOMA-AC                       | `momaac`     | Off-policy | TD3-style multi-objective actor-critic      |
| Outer-loop MOMAPPO            | `mappo`      | On-policy  | Outer-loop, single-preference baseline      |
| MASAC                         | `masac`      | Off-policy | Multi-agent SAC baseline                    |
| IP                            | `ippo`       | On-policy  | Independent PPO baseline                    |


### Main experiments

The main experiments use the Hydra entry point:

```powershell
python benchmarl/run.py `
  algorithm=cmomappo `
  task=sar/moroom128cpu `
  experiment=sar/cms.yaml `
  seed=0
```

The following PowerShell example runs a common grid over algorithms, tasks, and random seeds:

```powershell
$algorithms = @("cmomappo", "pcma", "momaac", "mappo", "masac", "ippo")
$tasks = @("moroom128cpu", "momaze128cpu", "moempty128cpu")
$seeds = @(0, 42, 3408)

foreach ($algorithm in $algorithms) {
  foreach ($task in $tasks) {
    foreach ($seed in $seeds) {
      python benchmarl/run.py `
        algorithm=$algorithm `
        task="sar/$task" `
        experiment=sar/cms.yaml `
        seed=$seed
    }
  }
}
```

The main settings in `benchmarl/conf/experiment/sar/cms.yaml` are:


| Parameter                        | Value                                               |
| -------------------------------- | --------------------------------------------------- |
| Total environment steps          | 1,000,000                                           |
| Learning rate                    | `5e-5`                                              |
| Discount factor                  | `0.99`                                              |
| Collection / replay device       | CPU / CPU                                           |
| Training device                  | CUDA                                                |
| Parallel collection environments | 8                                                   |
| Frames collected per batch       | 4,000                                               |
| On-policy minibatches            | 500 × 15 epochs                                     |
| Off-policy updates               | 300 per batch, batch size 256                       |
| Replay buffer                    | 250,000; earlier saved runs used 200,000 or 300,000 |
| Evaluation interval              | 20,000 steps                                        |
| Evaluation workload              | 8 episodes × 8 rollout rounds                       |
| Checkpoints                      | Every 200,000 steps; keep the latest 2              |
| Logging                          | CSV + W&B, `project="aaai"`                         |


On resource-constrained machines, use `scripts/train_sar_manual.py --serial`, or reduce `on_policy_n_envs_per_worker`, `off_policy_n_envs_per_worker`, and the evaluation parallelism.

### Output layout

By default, Hydra writes training outputs to:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra/
├── <algorithm>_<task>_<model>__<run-id>/
│   ├── checkpoints/checkpoint_<frames>.pt
│   ├── config.pkl
│   ├── scalars/
│   └── texts/hparams0.txt
└── <hydra log>
```

Keep `config.pkl` and its checkpoint in the same relative directory layout. The transfer and generalization scripts locate `config.pkl` two directory levels above the checkpoint.

## 6. Evaluation metrics

The metrics are implemented in `benchmarl/evaluations/mo_metrics.py`.


| Abbreviation | Metric                     | Direction | Computation                                                                                |
| ------------ | -------------------------- | --------- | ------------------------------------------------------------------------------------------ |
| GEU          | Global Expected Utility    | ↑         | `eval/agents/reward/episode_reward_mean`                                                   |
| C            | Pareto Cardinality         | ↑         | Number of non-dominated points                                                             |
| HV           | Normalized Hypervolume     | ↑         | Min-max normalized, with reference point `[-0.05, -0.05]`                                  |
| S            | Schott Sparsity            | ↓         | Non-uniformity of the Pareto-point distribution                                            |
| PAS          | Preference Alignment Score | ↑         | Spearman correlation between preference `u` and normalized arc-length rank along the front |


The primary W&B keys are:

```text
eval/agents/reward/episode_reward_mean
evaluation/mo/hypervolume
evaluation/mo/cardinality
evaluation/mo/sparsity
evaluation/mo/n_episodes
evaluation/mo/table
```

The Pareto table contains at least:

```text
preference_u, exploration_ratio, neg_path_length
```



## 7. Exporting W&B data and plotting results

Export the aggregated learning-curve CSV files and Pareto-table CSV files from W&B. The current scripts expect the following filenames:


| Scenario | GEU           | C           | HV            | Pareto front              |
| -------- | ------------- | ----------- | ------------- | ------------------------- |
| Room     | `guroom.csv`  | `croom.csv` | `hvroom.csv`  | `pareto fronter room.csv` |
| Maze     | `guemaze.csv` | `cmaze.csv` | `nhvmaze.csv` | `pareto front maze.csv`   |


Place the CSV files under `scripts/`. The learning-curve scripts multiply W&B's iteration `Step` by 4,000 to recover the number of environment steps. Columns ending in `__MIN` and `__MAX` are used as the lower and upper bounds of the shaded region.

Generate the four-panel Room figure:

```powershell
python scripts/combine_single_column.py `
  --room `
  --output outputs/room_gu_c_hv_pareto_1x4.pdf
```

Generate the four-panel Maze figure:

```powershell
python scripts/combine_single_column.py `
  --output outputs/gu_c_hv_pareto_1x4.pdf
```

Plot individual panels:

```powershell
python scripts/plot_gu.py --csv scripts/guroom.csv --output outputs/room_gu.pdf
python scripts/plot_c.py --csv scripts/croom.csv --output outputs/room_c.pdf
python scripts/plot_hv.py --csv scripts/hvroom.csv --output outputs/room_hv.pdf
python scripts/plot_pareto_front.py `
  --csv "scripts/pareto fronter room.csv" `
  --output outputs/pareto_front_room.pdf
```

Compute PAS:

```powershell
python scripts/compute_pas.py "scripts/pas room.csv"
```

Generate other paper figures:

```powershell
python scripts/plot_momaexplore_motivation.py
python scripts/plot_pareto_comparison_room_maze.py
python scripts/plot_room_pcma_cmomappo_pareto.py
python scripts/plot_room_pcma_cmomappo_hv_pas_diagnostic.py
```

Preview of the current main Room results:

Room main results

## 8. Cross-agent-count transfer

Load actor weights from a four-agent Room checkpoint into three-agent and two-agent environments. The `--actor-only` option copies only actor parameters with matching names and shapes, avoiding incompatible critic parameters or agent-count-dependent state.

```powershell
python scripts/evaluate_sar_cross_agents.py `
  "outputs/<date>/<time>/<run>/checkpoints/checkpoint_1000000.pt" `
  --tasks moroom128cpu3a moroom128cpu2a `
  --episodes 64 `
  --actor-only `
  --output-dir "outputs/cross_agent_eval/<algorithm>_seed_<run-id>"
```

Aggregate transfer results for PCMAPPO, PCMA, and MOMA-AC:

```powershell
python scripts/aggregate_cross_agent_transfer.py `
  --base-dir outputs/cross_agent_eval `
  --output outputs/cross_agent_eval/cross_agent_transfer_table.tex
```

The evaluation directory contains `*_summary.json`, `*_episodes.csv`, CSV logger results for each task, and a combined `combined_summary.json`.

## 9. Unseen-map generalization

The same evaluation script transfers a four-agent Room actor to the Random, Den, and Den3 maps:

```powershell
python scripts/evaluate_sar_cross_agents.py `
  "outputs/<date>/<time>/<run>/checkpoints/checkpoint_1000000.pt" `
  --tasks morandom128cpu moden128cpu moden3128cpu `
  --episodes 64 `
  --actor-only `
  --output-dir "outputs/map_generalization/<algorithm>_seed_<run-id>"
```

The saved three-algorithm summaries are located at:

```text
outputs/map_generalization/map_generalization_3seed_raw.csv
outputs/map_generalization/map_generalization_3seed_summary.json
outputs/map_generalization/map_generalization_3seed_table.tex
```

The repository retains these aggregated artifacts, but not the standalone script that generated them. Individual evaluation results remain fully reproducible with `evaluate_sar_cross_agents.py`.

## 10. Summary of saved results

The following values come directly from the aggregated results saved in the repository and are reported as mean ± sample standard deviation.

### Room: transfer of a four-agent actor across agent counts


| Setting | Algorithm | HV ↑            | C ↑              | GEU ↑            | PAS ↑           |
| ------- | --------- | --------------- | ---------------- | ---------------- | --------------- |
| Room 4A | PCMAPPO   | 0.66 ± 0.08     | 28.67 ± 2.00     | 1.72 ± 0.87      | 0.91 ± 0.03     |
| Room 4A | PCMA      | **0.78 ± 0.03** | 21.33 ± 1.50     | **2.06 ± 0.05**  | 0.62 ± 0.17     |
| Room 4A | MOMA-AC   | 0.39 ± 0.08     | 27.00 ± 4.50     | 0.80 ± 0.58      | 0.60 ± 0.24     |
| Room 3A | PCMAPPO   | 0.49 ± 0.05     | 6.00 ± 1.00      | -7.38 ± 1.07     | -0.02 ± 0.16    |
| Room 3A | PCMA      | **0.61 ± 0.24** | **24.67 ± 6.43** | **-1.03 ± 0.94** | **0.62 ± 0.18** |
| Room 3A | MOMA-AC   | 0.30 ± 0.02     | 19.67 ± 4.51     | -2.19 ± 1.12     | 0.52 ± 0.30     |
| Room 2A | PCMAPPO   | **0.44 ± 0.03** | 11.00 ± 1.73     | -5.73 ± 5.06     | -0.08 ± 0.10    |
| Room 2A | PCMA      | 0.41 ± 0.08     | **21.67 ± 3.06** | **-0.90 ± 0.32** | **0.55 ± 0.13** |
| Room 2A | MOMA-AC   | 0.24 ± 0.01     | 17.00 ± 1.73     | -2.45 ± 2.25     | 0.52 ± 0.16     |


These results show that PCMA maintains relatively high Pareto cardinality, GEU, and preference alignment after the agent count changes. PCMAPPO achieves the highest HV in the two-agent Room setting, but its C, GEU, and PAS decrease substantially.

### Map-generalization overview

- **Random~128:** PCMAPPO has the highest mean HV (`0.71`), while PCMA has the highest C (`29.67`) and a relatively high PAS (`0.66`).
- **Den~128:** PCMAPPO has the highest mean HV (`0.64`), while PCMA has the highest PAS (`0.67`).
- **Den3~128:** PCMAPPO has the highest mean HV (`0.63`), while MOMA-AC has the highest C (`31.33`) and PAS (`0.45`).
- See `outputs/map_generalization/map_generalization_3seed_summary.json` for all values and per-run results.



## 11. Reproducibility notes

1. **The recorded random seeds are not fully consistent.** Among the main saved Room artifacts, PCMA uses seeds `0/42/1234`, MOMA-AC uses `0/42/3408`, and the three saved PCMAPPO runs use `0/0/3408`, including a duplicate seed 0. The existing PCMAPPO aggregate should therefore be described as three runs, not three unique seeds. For new experiments, use the same set of unique seeds for every algorithm.
2. **Historical hyperparameters vary.** Saved runs use replay-buffer sizes of 200k, 250k, and 300k. For paper results, treat each run's `texts/hparams0.txt` and `config.pkl` as authoritative.
3. **Determinism settings vary.** The main training configuration uses `evaluation_deterministic_actions: false`, while some post-processing evaluation artifacts used deterministic/static evaluation. Use the same setting when comparing algorithms.
4. **Unity worker ports can conflict.** If the environment fails to reconnect after an abnormal exit, terminate any remaining Unity environment processes before running a serial smoke test.
5. **Windows paths are machine-specific.** When sharing a checkpoint, also share its task YAML, `config.pkl`, and normalization anchors.



## 12. Repository guide


| Path                                     | Description                                                         |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `benchmarl/environments/sar/`            | Unity wrapper, task classes, and preference sampling                |
| `benchmarl/algorithms/`                  | PCMA, PCMAPPO, MOMA-AC/SAC/GPI-PD, and other algorithms             |
| `benchmarl/conf/task/sar/`               | Room, Maze, Empty, transfer, and generalization task configurations |
| `benchmarl/conf/algorithm/`              | Algorithm hyperparameters                                           |
| `benchmarl/conf/experiment/sar/cms.yaml` | Main experiment configuration                                       |
| `benchmarl/evaluations/mo_metrics.py`    | Pareto, HV, C, and S metrics                                        |
| `scripts/train_sar_manual.py`            | Training entry point without Hydra                                  |
| `scripts/evaluate_sar_cross_agents.py`   | Actor-only evaluation across agent counts and maps                  |
| `scripts/`                               | W&B CSV files, PAS computation, and paper plotting scripts          |
| `outputs/`                               | Hydra outputs, evaluation summaries, and generated figures          |




## Acknowledgments and license

The training framework is based on BenchMARL, environment communication uses Unity ML-Agents, and the reinforcement-learning backend uses TorchRL. See `LICENSE` for the original BenchMARL license and the license retained by this repository.