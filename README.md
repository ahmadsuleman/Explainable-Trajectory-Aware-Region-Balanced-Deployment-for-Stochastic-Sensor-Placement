# TARBD: Trajectory-Aware Region-Balanced Sensor Deployment

> Explainable reinforcement learning and region-aware greedy baselines for stochastic trajectory sensor placement.

TARBD, **Trajectory-Aware Region-Balanced Deployment**, is a simulation framework for placing a limited number of sensors in a surveillance region where targets move along stochastic trajectories. The project extends static void-probability sensor placement into a **movement-constrained sequential deployment problem** using a DQN policy, explainable KPI rewards, and region-aware residual-risk metrics.

The core idea is simple: instead of only maximizing global detection, the agent should also avoid leaving important trajectory regions poorly covered.

---

## Key Features

- **Trajectory-aware modeling**
  - Generates stochastic linear trajectories in a `10 km x 10 km` arena.
  - Maps each trajectory into representation space `l = (alpha, p)` using normal-form line geometry.

- **MAP-smoothed Poisson intensity estimation**
  - Estimates trajectory density over `(alpha, p)` using a smoothed Poisson log-intensity approximation.
  - Uses this intensity as the expected traffic field for future target trajectories.

- **RL-compatible deployment environment**
  - Sequential movement and placement actions.
  - Continuous state features with residual-risk map, placed sensors, and agent position.
  - Gymnasium-compatible wrapper for Stable-Baselines3 DQN.

- **Explainable reward design**
  - Rewards reduction in expected undetected trajectory mass `U(S)`.
  - Rewards improvement in worst-region detection.
  - Penalizes invalid placement, movement cost, and redundant sensor placement.

- **Baselines and evaluation**
  - Random sequential policy.
  - One-step sequential greedy policy.
  - Static global greedy placement upper bound.
  - Static region-aware greedy placement upper bound.

- **Publication-ready outputs**
  - CSV metric summaries.
  - Training curves.
  - Physical sensor layout plots.
  - 3D intensity, detection, and residual-risk surfaces.

---

## Project Motivation

Traditional void-probability-based placement chooses a fixed set of sensors offline. That is useful, but it does not model how a deployment platform moves through the environment or how each placement decision affects regional risk.

TARBD reframes the problem as a sequential decision process:

1. Generate or observe historical target trajectories.
2. Estimate the stochastic trajectory intensity field.
3. Move a deployment agent through the arena.
4. Place sensors under budget and spacing constraints.
5. Optimize for both global missed-trajectory reduction and worst-region risk reduction.

This makes each action interpretable through metrics such as expected undetected mass, void probability, detection score, residual risk, and regional balance.

---

## Method Overview

### 1. Trajectory Representation

Each linear trajectory is represented as:

```text
l = (alpha, p)
```

where:

- `alpha` is the normal angle of the trajectory line.
- `p` is the signed perpendicular offset from the arena center.

The normal-form line equation is:

```text
x cos(alpha) + y sin(alpha) = p
```

This avoids slope singularities and supports exact sensor-to-trajectory distance computation.

### 2. Sensor Detection Model

For a trajectory type `l` and sensor `s`, the detection probability is:

```text
gamma(l, s) = rho * exp(-d(l, s)^2 / (2 * sigma^2))
```

where:

- `rho` is the maximum detection probability.
- `sigma` controls sensing spread.
- `d(l, s)` is the perpendicular distance from the sensor to the trajectory line.

For a sensor set `S`, the miss probability is:

```text
pi(l, S) = product over sensors [1 - gamma(l, s)]
```

### 3. Core Metrics

| Metric | Meaning |
|---|---|
| `U(S)` | Expected undetected trajectory mass |
| `V(S) = exp(-U(S))` | Void probability |
| `logV(S) = -U(S)` | Log-void value |
| `D(S)` | Global detection score |
| `R(l,S)` | Residual-risk map |
| `U_k(S)` | Regional residual risk |
| `U_max(S)` | Worst regional residual risk |
| `D_min(S)` | Worst-region detection score |
| `risk_imbalance_cv` | Coefficient of variation of active regional residual risk |

### 4. Sequential MDP

The deployment agent uses five actions:

| Action ID | Action |
|---:|---|
| `0` | Move north |
| `1` | Move south |
| `2` | Move east |
| `3` | Move west |
| `4` | Place sensor |

Each episode ends when the sensor budget is used or the maximum step count is reached.

---

## Repository Structure

Recommended cleaned file names:

```text
.
├── DQN_compatible_env.py        # Core trajectory generator, intensity estimator, and RL environment
├── DQN_traing.py                # DQN training, evaluation, plots, and reports
├── baselines_greedy_belmen.py   # Mechanism audit and greedy baseline comparison
├── env_visualizer.py            # Explanatory plots for trajectories, state space, reward, and sensing model
├── paper.tex                    # Paper source for TARBD
└── README.md
```

If your downloaded files include names such as `DQN_compatible_env(1).py`, rename them before running:

```bash
mv "DQN_compatible_env(1).py" DQN_compatible_env.py
mv "DQN_traing(1).py" DQN_traing.py
mv "baselines_greedy_belmen(1).py" baselines_greedy_belmen.py
mv "Pasted text(214).txt" paper.tex
```

The training and visualization scripts import `DQN_compatible_env`, so the environment file should be named exactly `DQN_compatible_env.py` unless you update the imports.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install "stable-baselines3[extra]" gymnasium torch pandas matplotlib scipy numpy
```

For a lighter setup without DQN training, the environment and greedy audit mainly need:

```bash
pip install numpy scipy matplotlib pandas
```

---

## Quick Start

### Run the DQN training and evaluation

```bash
python DQN_traing.py \
  --total-timesteps 100000 \
  --eval-episodes 30 \
  --output-dir dqn_sensor_outputs
```

For a longer experiment:

```bash
python DQN_traing.py \
  --total-timesteps 1000000 \
  --eval-episodes 30 \
  --traffic-type multi_corridor \
  --max-sensors 4 \
  --max-steps 30 \
  --output-dir dqn_sensor_outputs
```

### Run the region-aware maturity audit

```bash
python baselines_greedy_belmen.py \
  --output-dir region_aware_maturity_outputs
```

### Generate explanatory environment figures

```bash
python env_visualizer.py
```

---

## Main Outputs

### DQN evaluation outputs

The DQN script writes results to `dqn_sensor_outputs/`.

| File | Description |
|---|---|
| `run_config.json` | Full experiment configuration |
| `dqn_sensor_policy.zip` | Trained Stable-Baselines3 DQN model |
| `monitor.csv` | Stable-Baselines3 monitor log |
| `training_episode_rewards.csv` | Episode reward history |
| `evaluation_episode_detail.csv` | Per-episode evaluation metrics |
| `evaluation_summary.csv` | Aggregate comparison across methods |
| `REPORT.md` | Compact generated experiment report |
| `01_dqn_training_curve.png` | Training reward curve |
| `07_physical_layout_comparison.png` | Sensor layout comparison |
| `08_dqn_final_residual_risk_3d.png` | Final DQN residual-risk surface |
| `09_paper_static_residual_risk_3d.png` | Static global greedy residual-risk surface |
| `10_region_aware_static_residual_risk_3d.png` | Static region-aware residual-risk surface |

### Baseline audit outputs

The maturity audit writes results to `region_aware_maturity_outputs/`.

| File | Description |
|---|---|
| `mechanism_audit.json` | Mechanism correctness checks |
| `baseline_metrics.csv` | Random, global greedy, and region-aware greedy metrics |
| `AUDIT_REPORT.md` | Human-readable audit report |
| `01_intensity_surface_3d.png` | Estimated trajectory intensity |
| `02_center_sensor_detection_surface_3d.png` | One-sensor detection surface |
| `03_initial_residual_risk_3d.png` | Initial residual-risk surface |
| `04_paper_greedy_residual_risk_3d.png` | Global greedy residual risk |
| `05_region_aware_residual_risk_3d.png` | Region-aware residual risk |
| `06_first_sensor_global_value_3d.png` | First-sensor global objective surface |
| `07_first_sensor_region_aware_value_3d.png` | First-sensor region-aware objective surface |
| `08_metric_comparison_bar_chart.png` | Metric comparison plot |
| `09_physical_sensor_layout_comparison.png` | Physical layout comparison |

---

## Reproducing the Paper Setup

Default settings match the main simulation setup:

| Parameter | Default |
|---|---:|
| Arena size | `10.0 km x 10.0 km` |
| Traffic type | `multi_corridor` |
| Number of trajectories | `350` |
| Representation grid | `30 x 30` |
| Expected traffic | `10.0` |
| Max sensors | `4` |
| Max steps | `30` |
| Motion step | `1.0 km` |
| Maximum detection probability `rho` | `0.95` |
| Sensor spread `sigma` | `0.45` in DQN script, `0.65` in baseline audit |
| Minimum sensor spacing | `0.5 km` |

The paper reports that TARBD improves over random sequential deployment by reducing expected undetected mass from `7.409` to `6.860`, increasing void probability from `7.21e-4` to `1.05e-3`, improving global detection from `0.259` to `0.314`, and reducing worst regional residual risk from `0.481` to `0.398`.

Static greedy methods can still score higher in final placement quality because they are not movement-constrained. Treat them as upper bounds, not direct deployment-policy competitors.

---

## Evaluation Methods

| Method | Movement-constrained | Learns policy | Purpose |
|---|---:|---:|---|
| `random_sequential` | Yes | No | Lower baseline |
| `one_step_greedy_sequential` | Yes | No | Strong sequential heuristic |
| `dqn_sequential` | Yes | Yes | Main deployable TARBD policy |
| `Static-Greedy` | No | No | Global greedy upper bound |
| `Region_Aware_Greedy` | No | No | Region-aware static upper bound |

Use `dqn_sequential` mainly against `random_sequential` and `one_step_greedy_sequential`, because all three obey the same movement and episode constraints.

---

## Important Notes

- This project uses a **MAP-smoothed Poisson log-intensity approximation**, not full R-INLA.
- The environment is a controlled toy simulator, designed for mechanism consistency and method evaluation.
- Static greedy baselines are useful for placement-quality comparison, but they do not represent deployable sequential policies.
- If DQN underuses the full sensor budget, tune terminal unplaced penalties, placement reward, exploration settings, or maximum steps.
- Random seeds are exposed in the CLI for reproducible experiments.

---

## Useful CLI Examples

### Faster smoke test

```bash
python DQN_traing.py \
  --total-timesteps 10000 \
  --eval-episodes 5 \
  --output-dir smoke_test_outputs
```

### Test a dispersed traffic scenario

```bash
python DQN_traing.py \
  --traffic-type dispersed \
  --total-timesteps 100000 \
  --eval-episodes 30 \
  --output-dir dispersed_outputs
```

### Increase the sensor budget

```bash
python DQN_traing.py \
  --max-sensors 6 \
  --max-steps 45 \
  --total-timesteps 200000 \
  --output-dir six_sensor_outputs
```

### Strengthen regional balancing in static greedy

```bash
python DQN_traing.py \
  --static-w-global 1.0 \
  --static-w-worst-det 3.0 \
  --static-w-umax 3.0 \
  --static-w-balance 1.5 \
  --output-dir stronger_region_static_outputs
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'DQN_compatible_env'`

Rename the uploaded environment file:

```bash
mv "DQN_compatible_env(1).py" DQN_compatible_env.py
```

or update imports in the scripts.

### Stable-Baselines3 import error

Install the RL dependencies:

```bash
pip install "stable-baselines3[extra]" gymnasium torch
```

### Training is slow

Start with fewer timesteps:

```bash
python DQN_traing.py --total-timesteps 10000 --eval-episodes 5
```

Then scale to `100000` or `1000000` once the pipeline is working.

### DQN places too few sensors

Try one or more of these changes:

```bash
--terminal-unplaced-penalty 0.8
--placement-cost 0.0
--max-steps 45
--exploration-final-eps 0.10
```

---

## Citation

If you use this work, cite the associated paper:

```bibtex
@article{suleman2026tarbd,
  title   = {Explainable Trajectory-Aware Region-Balanced Deployment for Stochastic Sensor Placement},
  author  = {Suleman, Ahmad},
  year    = {2026},
  note    = {Trajectory-aware sequential sensor deployment with DQN and region-balanced residual-risk control}
}
```

---

## License

No license file was included with the submitted code. Add a license before public release, for example MIT for open-source research code or a restricted academic-use license if needed.

---

## Author

**Ahmad Suleman**

Project: **Explainable Trajectory-Aware Region-Balanced Deployment for Stochastic Sensor Placement**
