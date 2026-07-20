MOMACoverage title

## Overview 🧐

MOMACoverage is a MOMARL benchmark for continuous cooperative multi-agent exploration with two intrinsic conflicting objectives, map coverage and path length. Agents in MOMACoverage need to cooperatively maximize environment coverage while minimizing the mean path length. 

MOMACoverage is build upon on [BenchMARL](https://github.com/facebookresearch/BenchMARL) 1.5.1 with a Unity [ML-Agents toolkit](https://github.com/unity-technologies/ml-agents).

MOMACoverage Overview

## 1️⃣ Quick Installation

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

# Install it locally to access the configs and scripts
pip install -e BenchMARL
```



## 2️⃣ Downlaod Maps and Add Path to Yaml file

Unity builds are not included in this repository, please download `maps` from the following link: [Google Drive](https://drive.google.com/drive/folders/1SGUg8nzm18HtZm1xMEH6AjKDcpDwedAp?usp=drive_link). 

Open `benchmarl/conf/task/sar/xxx.yaml` and update the env_path field:

```yaml
env_path:  fill the downloaded env path, such as "E:\\PythonCode\\MOCMSENV\\MOROOM128CPU\\UnityEnvironment"
```

We provide a Windows executable, allowing you to experience the benchmark directly by simply launching the .exe file. Linux users could download the corresponding Linux verison and run the provided x86_64 executable.

### Current Map and Task in MAMOCoverage


| Experiment         | Hydra task       | Agents | Normalization anchors    |
| ------------------ | ---------------- | ------ | ------------------------ |
| Room Map          | `moroom128cpu`   | 4      | `[[105, 0], [0, -1100]]` |
| Maze Map          | `momaze128cpu`   | 4      | `[[105, 0], [0, -1000]]` |
| Empty Map         | `moempty128cpu`  | 4      | `[[105, 0], [0, -1100]]` |
| Random Map | `morandom128cpu` | 4      | `[[105, 0], [0, -1100]]` |
| Den Map | `moden128cpu`    | 4      | `[[105, 0], [0, -1100]]` |
| Den3 Map | `moden3128cpu`   | 4      | `[[105, 0], [0, -1100]]` |


## 3️⃣ Training
The main experiments use the Hydra entry point:

```
python benchmarl/run.py `
  algorithm=cmomappo `
  task=sar/moroom128cpu `
  experiment=sar/cms.yaml `
  seed=0
```

### MOMARL and MARL in MAMOCoverage
MOMAcoverage supports both MARL and MOMARL algorithm, and has intergrated several classical MOMARL algorithm into framework. 
| Name used in the README/paper | `algorithm=` | Paradigm | Type       | Description                                 |
| ----------------------------- | ------------ | -------- | ---------- | ------------------------------------------- |
| PCMAPPO                       | `cmomappo`   | MOMARL   | On-policy  | Preference-conditioned MAPPO                |
| PCMA                          | `pcma`       | MOMARL   | On-policy  | Global/local preference coordination policy |
| MOMA-AC                       | `momaac`     | MOMARL   | Off-policy | TD3-style multi-objective actor-critic      |
| Outer-loop MOMAPPO            | `mappo`      | MOMARL   | On-policy  | Outer-loop, single-preference baseline      |
| MASAC                         | `masac`      | MARL     | Off-policy | Multi-agent SAC baseline                    |
| IPPO                          | `ippo`       | MARL     | On-policy  | Independent PPO baseline                    |


## 4️⃣  Evaluation metrics

The metrics are implemented in `benchmarl/evaluations/mo_metrics.py`.

| Abbreviation | Metric                     | Direction | Computation                                                                                |
| ------------ | -------------------------- | --------- | ------------------------------------------------------------------------------------------ |
| GEU          | Global Expected Utility    | ↑         | `eval/agents/reward/episode_reward_mean`                                                   |
| C            | Pareto Cardinality         | ↑         | Number of non-dominated points                                                             |
| HV           | Normalized Hypervolume     | ↑         | Min-max normalized, with reference point `[-0.05, -0.05]`                                  |
| PAS          | Preference Alignment Score | ↑         | Spearman correlation between preference `u` and normalized arc-length rank along the front |


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

## Acknowledgments and license

The training framework is based on BenchMARL, environment communication uses Unity ML-Agents, and the reinforcement-learning backend uses TorchRL. See `LICENSE` for the original BenchMARL license and the license retained by this repository.