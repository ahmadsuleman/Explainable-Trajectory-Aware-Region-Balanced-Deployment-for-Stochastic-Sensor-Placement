"""
train_test_dqn_sb3.py

Stable-Baselines3 DQN training and evaluation script for the cleaned
trajectory-aware sensor deployment environment.

What this script does
---------------------
1. Builds the toy trajectory-intensity environment.
2. Wraps it as a Gymnasium environment compatible with Stable-Baselines3.
3. Trains a DQN policy on the sequential movement-constrained deployment task.
4. Evaluates the trained DQN against:
      - random sequential policy,
      - one-step sequential greedy policy,
      - paper-style static global greedy placement upper bound,
      - region-aware static greedy placement upper bound.
5. Exports CSV summaries, plots, and the trained model.

Install dependencies
--------------------
pip install "stable-baselines3[extra]" gymnasium torch pandas matplotlib scipy

Example
-------
python train_test_dqn_sb3.py \
    --total-timesteps 100000 \
    --eval-episodes 30 \
    --output-dir dqn_sensor_outputs

Notes
-----
The static greedy baselines are not movement-constrained. They are useful upper
bounds for placement quality. DQN should be judged mainly against sequential
random and sequential one-step greedy under the same movement constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The cleaned environment must be in the same directory or on PYTHONPATH.
from DQN_compatible_env import (
    IntensityEstimate,
    MAPLGCPEstimator,
    TrajectoryGenerator,
    TrajectorySensorDeploymentEnv,
)


# =============================================================================
# Stable-Baselines3 imports are delayed so the script can print a useful message.
# =============================================================================

def import_sb3():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Missing Stable-Baselines3/Gymnasium dependencies. Install with:\n"
            "    pip install 'stable-baselines3[extra]' gymnasium torch pandas matplotlib scipy\n"
        ) from exc

    return gym, spaces, DQN, BaseCallback, Monitor, set_random_seed


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def flatten_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the environment dict observation into a single float32 vector."""
    parts = [
        np.asarray(obs["features"], dtype=np.float32).reshape(-1),
        np.asarray(obs["residual_map"], dtype=np.float32).reshape(-1),
        np.asarray(obs["sensor_locations"], dtype=np.float32).reshape(-1),
        np.asarray(obs["agent_xy"], dtype=np.float32).reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32)


def region_imbalance_cv(metrics: Dict[str, Any]) -> float:
    """Coefficient of variation of active regional residual risk."""
    region_u = np.asarray(metrics.get("region_U", []), dtype=float)
    active = np.asarray(metrics.get("active_region_mask", np.ones_like(region_u, dtype=bool)), dtype=bool)
    values = region_u[active]
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def metrics_to_row(method: str, metrics: Dict[str, Any], sensors: np.ndarray, episode_reward: Optional[float] = None) -> Dict[str, Any]:
    return {
        "method": method,
        "episode_reward": np.nan if episode_reward is None else float(episode_reward),
        "num_sensors": int(len(sensors)),
        "U_expected_undetected": float(metrics["expected_undetected"]),
        "V_void_probability": float(metrics["void_probability"]),
        "D_global_detection": float(metrics["detection_score"]),
        "D_min_worst_region_detection": float(metrics["worst_region_detection"]),
        "U_max_worst_region_residual": float(metrics["worst_region_U"]),
        "D_mean_active_region_detection": float(metrics["mean_region_detection"]),
        "risk_imbalance_cv": region_imbalance_cv(metrics),
        "dominant_alpha_deg": float(np.rad2deg(metrics["dominant_alpha"])),
        "dominant_p_km": float(metrics["dominant_p"]),
        "sensors": json.dumps(np.asarray(sensors, dtype=float).round(4).tolist()),
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = [
        "episode_reward",
        "num_sensors",
        "U_expected_undetected",
        "V_void_probability",
        "D_global_detection",
        "D_min_worst_region_detection",
        "U_max_worst_region_residual",
        "D_mean_active_region_detection",
        "risk_imbalance_cv",
        "dominant_alpha_deg",
        "dominant_p_km",
    ]
    grouped = []
    for method, g in df.groupby("method", sort=False):
        row = {"method": method, "episodes": int(len(g))}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0)) if len(g) > 1 else 0.0
        grouped.append(row)
    return pd.DataFrame(grouped)


# =============================================================================
# Build environment
# =============================================================================

def build_intensity(args: argparse.Namespace) -> Tuple[IntensityEstimate, Any]:
    generator = TrajectoryGenerator(arena_size=args.arena_size, seed=args.data_seed)
    data = generator.generate(n=args.n_trajectories, traffic_type=args.traffic_type)

    estimator = MAPLGCPEstimator(
        n_alpha=args.n_alpha,
        n_p=args.n_p,
        p_min=-args.arena_size / 2.0,
        p_max=args.arena_size / 2.0,
        tau=args.tau,
        kappa=args.kappa,
    )
    intensity = estimator.fit(data.rep_points)
    return intensity, data


def make_core_env(intensity: IntensityEstimate, args: argparse.Namespace, seed: int) -> TrajectorySensorDeploymentEnv:
    reward_weights = {
        "risk_reduction_norm": args.w_risk,
        "worst_detection_gain": args.w_worst_detection,
        "step_cost": args.step_cost,
        "move_cost": args.move_cost,
        "invalid": args.invalid_cost,
        "placement_cost": args.placement_cost,
        "redundancy": args.redundancy_cost,
        "terminal_detection": args.terminal_detection,
        "terminal_worst_detection": args.terminal_worst_detection,
        "terminal_unplaced_penalty": args.terminal_unplaced_penalty,
    }
    return TrajectorySensorDeploymentEnv(
        intensity=intensity,
        arena_size=args.arena_size,
        expected_traffic=args.expected_traffic,
        max_sensors=args.max_sensors,
        max_steps=args.max_steps,
        motion_step=args.motion_step,
        rho=args.rho,
        sigma=args.sigma,
        min_sensor_spacing=args.min_sensor_spacing,
        reward_weights=reward_weights,
        seed=seed,
    )


# =============================================================================
# Gymnasium wrapper for SB3
# =============================================================================

def make_gym_wrapper_class():
    gym, spaces, *_ = import_sb3()

    class FlatSensorDeploymentGymEnv(gym.Env):
        """Gymnasium wrapper that flattens dict observations for DQN MlpPolicy."""

        metadata = {"render_modes": []}

        def __init__(self, intensity: IntensityEstimate, args: argparse.Namespace, seed: int = 0):
            super().__init__()
            self.intensity = intensity
            self.args = args
            self.seed_value = seed
            self.core = make_core_env(intensity, args, seed=seed)

            sample = flatten_observation(self.core.get_observation())
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=sample.shape,
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(self.core.n_actions)

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
            super().reset(seed=seed)
            if seed is not None:
                self.seed_value = int(seed)
            obs = self.core.reset(seed=self.seed_value if seed is not None else None)
            return flatten_observation(obs), {}

        def step(self, action: int):
            obs, reward, terminated, truncated, info = self.core.step_gym(int(action))
            return flatten_observation(obs), float(reward), bool(terminated), bool(truncated), info

    return FlatSensorDeploymentGymEnv


# =============================================================================
# Training callback
# =============================================================================

def make_training_callback_class():
    _, _, _, BaseCallback, *_ = import_sb3()

    class EpisodeStatsCallback(BaseCallback):
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episode_rewards: List[float] = []
            self.episode_lengths: List[int] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    self.episode_rewards.append(float(ep["r"]))
                    self.episode_lengths.append(int(ep["l"]))
            return True

    return EpisodeStatsCallback


# =============================================================================
# Sequential policy evaluation
# =============================================================================

def evaluate_dqn_model(model: Any, intensity: IntensityEstimate, args: argparse.Namespace, episodes: int, seed_offset: int) -> List[Dict[str, Any]]:
    FlatEnv = make_gym_wrapper_class()
    rows: List[Dict[str, Any]] = []

    for ep in range(episodes):
        env = FlatEnv(intensity, args, seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            done = bool(terminated or truncated)

        sensors = np.asarray(env.core.sensors, dtype=float)
        metrics = env.core.current_metrics
        rows.append(metrics_to_row("dqn_sequential", metrics, sensors, episode_reward=total_reward))

    return rows


def evaluate_random_sequential(intensity: IntensityEstimate, args: argparse.Namespace, episodes: int, seed_offset: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = make_core_env(intensity, args, seed=seed_offset + ep)
        env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        while not env.done:
            action = int(env.rng.integers(0, env.n_actions))
            _, reward, _, _ = env.step(action)
            total_reward += float(reward)
        sensors = np.asarray(env.sensors, dtype=float)
        rows.append(metrics_to_row("random_sequential", env.current_metrics, sensors, episode_reward=total_reward))
    return rows


def evaluate_one_step_greedy_sequential(intensity: IntensityEstimate, args: argparse.Namespace, episodes: int, seed_offset: int) -> List[Dict[str, Any]]:
    """Sequential baseline that greedily chooses the action with max immediate reward."""
    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = make_core_env(intensity, args, seed=seed_offset + ep)
        env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        while not env.done:
            candidates = env.evaluate_action_consequences()
            best = max(candidates, key=lambda r: r["reward"])
            _, reward, _, _ = env.step(int(best["action_id"]))
            total_reward += float(reward)
        sensors = np.asarray(env.sensors, dtype=float)
        rows.append(metrics_to_row("one_step_greedy_sequential", env.current_metrics, sensors, episode_reward=total_reward))
    return rows


# =============================================================================
# Static greedy baselines: upper bounds because they are not movement-constrained.
# =============================================================================

def candidate_grid(arena_size: float, grid_n: int) -> np.ndarray:
    xs = np.linspace(0.0, arena_size, grid_n)
    ys = np.linspace(0.0, arena_size, grid_n)
    return np.asarray([[x, y] for y in ys for x in xs], dtype=float)


def can_add_sensor(existing: List[np.ndarray], candidate: np.ndarray, max_sensors: int, min_spacing: float) -> bool:
    if len(existing) >= max_sensors:
        return False
    if len(existing) == 0:
        return True
    d = np.linalg.norm(np.asarray(existing) - candidate[None, :], axis=1)
    return float(np.min(d)) >= min_spacing


def static_greedy_baseline(
    intensity: IntensityEstimate,
    args: argparse.Namespace,
    method: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Static greedy placement baseline.

    method='Static-Greedy': maximizes global risk reduction.
    method='Region_Aware_Greedy': maximizes global + worst-region + imbalance terms.
    """
    env = make_core_env(intensity, args, seed=seed)
    env.reset(start_xy=[args.arena_size / 2.0, args.arena_size / 2.0], seed=seed)

    candidates = candidate_grid(args.arena_size, args.static_grid_n)
    sensors: List[np.ndarray] = []
    initial_metrics = env.compute_metrics([])
    initial_u = float(initial_metrics["expected_undetected"])
    initial_umax = float(initial_metrics["worst_region_U"])
    initial_cv = region_imbalance_cv(initial_metrics)

    for _ in range(args.max_sensors):
        old_metrics = env.compute_metrics(sensors)
        best_score = -np.inf
        best_candidate = None

        for cand in candidates:
            if not can_add_sensor(sensors, cand, args.max_sensors, args.min_sensor_spacing):
                continue

            sensors2 = [s.copy() for s in sensors] + [cand.copy()]
            new_metrics = env.compute_metrics(sensors2)
            d_u_norm = (old_metrics["expected_undetected"] - new_metrics["expected_undetected"]) / max(initial_u, 1e-12)

            if method == "Static-Greedy":
                score = d_u_norm
            elif method == "Region_Aware_Greedy":
                d_worst_det = new_metrics["worst_region_detection"] - old_metrics["worst_region_detection"]
                d_umax_norm = (old_metrics["worst_region_U"] - new_metrics["worst_region_U"]) / max(initial_umax, 1e-12)
                d_cv = region_imbalance_cv(old_metrics) - region_imbalance_cv(new_metrics)
                score = (
                    args.static_w_global * d_u_norm
                    + args.static_w_worst_det * d_worst_det
                    + args.static_w_umax * d_umax_norm
                    + args.static_w_balance * d_cv
                )
            else:
                raise ValueError(f"Unknown static greedy method: {method}")

            if score > best_score:
                best_score = float(score)
                best_candidate = cand.copy()

        if best_candidate is None:
            break
        sensors.append(best_candidate)

    final_metrics = env.compute_metrics(sensors)
    return metrics_to_row(method, final_metrics, np.asarray(sensors, dtype=float), episode_reward=None)


# =============================================================================
# Plots
# =============================================================================

def plot_training_curve(callback: Any, out_dir: Path) -> None:
    if not callback.episode_rewards:
        return
    rewards = np.asarray(callback.episode_rewards, dtype=float)
    window = min(50, max(1, len(rewards) // 5))
    if len(rewards) >= window:
        smooth = pd.Series(rewards).rolling(window=window, min_periods=1).mean().to_numpy()
    else:
        smooth = rewards

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.35, label="episode reward")
    ax.plot(np.arange(1, len(smooth) + 1), smooth, linewidth=2.0, label=f"rolling mean ({window})")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    # ax.set_title("DQN Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_dqn_training_curve.png", dpi=300)
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("D_global_detection_mean", "Global detection D", True),
        ("D_min_worst_region_detection_mean", "Worst-region detection D_min", True),
        ("U_expected_undetected_mean", "Expected undetected U", False),
        ("U_max_worst_region_residual_mean", "Worst-region residual U_max", False),
        ("risk_imbalance_cv_mean", "Risk imbalance CV", False),
    ]

    for idx, (col, title, higher_better) in enumerate(metrics, start=2):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        labels = summary["method"].tolist()
        values = summary[col].to_numpy(dtype=float)
        yerr_col = col.replace("_mean", "_std")
        yerr = summary[yerr_col].to_numpy(dtype=float) if yerr_col in summary else None
        ax.bar(labels, values, yerr=yerr, capsize=4)
        ax.set_ylabel(title)
        # ax.set_title(f"{title} ({'higher is better' if higher_better else 'lower is better'})")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:02d}_{col}.png", dpi=300)
        plt.close(fig)


def plot_residual_surface(env: TrajectorySensorDeploymentEnv, title: str, path: Path) -> None:
    R = env.current_metrics["residual_map"]
    X, Y = np.meshgrid(np.rad2deg(env.alpha_centers), env.p_centers, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, R, cmap="inferno", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("trajectory normal angle alpha (deg)")
    ax.set_ylabel("signed offset p (km)")
    ax.set_zlabel("residual risk R(alpha,p)")
    # ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.1, label="residual risk")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_physical_layout(data: Any, rows: List[Dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> None:
    selected_methods = [
        "Static-Greedy",
        "Region_Aware_Greedy",
        "dqn_sequential",
    ]
    fig, axes = plt.subplots(1, len(selected_methods), figsize=(5.2 * len(selected_methods), 5.2))
    if len(selected_methods) == 1:
        axes = [axes]

    by_method = {}
    for row in rows:
        if row["method"] not in by_method:
            by_method[row["method"]] = row

    for ax, method in zip(axes, selected_methods):
        for p1, p2 in data.segments[: min(len(data.segments), 450)]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.6, alpha=0.15)
        if method in by_method:
            sensors = np.asarray(json.loads(by_method[method]["sensors"]), dtype=float)
            if sensors.size > 0:
                ax.scatter(sensors[:, 0], sensors[:, 1], marker="^", s=130, label="Sensors")
        ax.set_xlim(0, args.arena_size)
        ax.set_ylim(0, args.arena_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x position (km)")
        ax.set_ylabel("y position (km)")
        ax.set_title(method)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "07_physical_layout_comparison.png", dpi=300)
    plt.close(fig)


def make_dqn_final_env(intensity: IntensityEstimate, args: argparse.Namespace, model: Any, seed: int) -> TrajectorySensorDeploymentEnv:
    FlatEnv = make_gym_wrapper_class()
    gym_env = FlatEnv(intensity, args, seed=seed)
    obs, _ = gym_env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = gym_env.step(int(action))
        done = bool(terminated or truncated)
    return gym_env.core


def make_static_final_env(intensity: IntensityEstimate, args: argparse.Namespace, row: Dict[str, Any], seed: int) -> TrajectorySensorDeploymentEnv:
    env = make_core_env(intensity, args, seed=seed)
    sensors = [np.asarray(s, dtype=float) for s in json.loads(row["sensors"])]
    env.sensors = sensors
    env.current_metrics = env.compute_metrics(sensors)
    return env


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test SB3 DQN on trajectory-aware sensor deployment.")

    # Output and seeds.
    parser.add_argument("--output-dir", type=str, default="dqn_sensor_outputs")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=1000)

    # Synthetic data and intensity.
    parser.add_argument("--traffic-type", type=str, default="multi_corridor", choices=["single_corridor", "multi_corridor", "dispersed"])
    parser.add_argument("--n-trajectories", type=int, default=350)
    parser.add_argument("--n-alpha", type=int, default=30)
    parser.add_argument("--n-p", type=int, default=30)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=1.2)

    # Environment.
    parser.add_argument("--arena-size", type=float, default=10.0)
    parser.add_argument("--expected-traffic", type=float, default=10.0)
    parser.add_argument("--max-sensors", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--motion-step", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--sigma", type=float, default=0.45)
    parser.add_argument("--min-sensor-spacing", type=float, default=0.5)

    # Environment reward weights for DQN and sequential greedy.
    parser.add_argument("--w-risk", type=float, default=1.0)
    parser.add_argument("--w-worst-detection", type=float, default=0.5)
    parser.add_argument("--step-cost", type=float, default=0.001)
    parser.add_argument("--move-cost", type=float, default=0.005)
    parser.add_argument("--invalid-cost", type=float, default=0.25)
    parser.add_argument("--placement-cost", type=float, default=0.0)
    parser.add_argument("--redundancy-cost", type=float, default=0.02)
    parser.add_argument("--terminal-detection", type=float, default=1.0)
    parser.add_argument("--terminal-worst-detection", type=float, default=0.5)
    parser.add_argument("--terminal-unplaced-penalty", type=float, default=0.4)

    # DQN hyperparameters.
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--target-update-interval", type=int, default=5000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--exploration-fraction", type=float, default=0.65)
    parser.add_argument("--exploration-final-eps", type=float, default=0.15)
    parser.add_argument("--policy-net", type=int, nargs="+", default=[256, 256])

    # Evaluation.
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--static-grid-n", type=int, default=31)
    parser.add_argument("--static-w-global", type=float, default=1.0)
    parser.add_argument("--static-w-worst-det", type=float, default=2.0)
    parser.add_argument("--static-w-umax", type=float, default=2.0)
    parser.add_argument("--static-w-balance", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir))
    save_json(vars(args), out_dir / "run_config.json")

    gym, spaces, DQN, BaseCallback, Monitor, set_random_seed = import_sb3()
    set_random_seed(args.train_seed)

    print("Building trajectory intensity...")
    intensity, data = build_intensity(args)

    print("Creating training environment...")
    FlatEnv = make_gym_wrapper_class()
    raw_train_env = FlatEnv(intensity, args, seed=args.train_seed)
    train_env = Monitor(raw_train_env, filename=str(out_dir / "monitor.csv"))

    callback_cls = make_training_callback_class()
    callback = callback_cls()

    policy_kwargs = {"net_arch": args.policy_net}

    print("Training DQN...")
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.train_seed,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(str(out_dir / "dqn_sensor_policy"))

    # Save training curve data.
    if callback.episode_rewards:
        pd.DataFrame(
            {
                "episode": np.arange(1, len(callback.episode_rewards) + 1),
                "reward": callback.episode_rewards,
                "length": callback.episode_lengths,
            }
        ).to_csv(out_dir / "training_episode_rewards.csv", index=False)

    print("Evaluating policies...")
    rows: List[Dict[str, Any]] = []
    rows.extend(evaluate_random_sequential(intensity, args, args.eval_episodes, args.eval_seed + 10000))
    rows.extend(evaluate_one_step_greedy_sequential(intensity, args, args.eval_episodes, args.eval_seed + 20000))
    rows.extend(evaluate_dqn_model(model, intensity, args, args.eval_episodes, args.eval_seed + 30000))

    # Static upper-bound baselines are single deterministic runs on the same intensity.
    paper_row = static_greedy_baseline(intensity, args, "Static-Greedy", seed=args.eval_seed + 40000)
    region_row = static_greedy_baseline(intensity, args, "Region_Aware_Greedy", seed=args.eval_seed + 50000)
    rows.append(paper_row)
    rows.append(region_row)

    detail = pd.DataFrame(rows)
    summary = summarize_rows(rows)
    detail.to_csv(out_dir / "evaluation_episode_detail.csv", index=False)
    summary.to_csv(out_dir / "evaluation_summary.csv", index=False)

    # Plots.
    print("Making plots...")
    plot_training_curve(callback, out_dir)
    plot_metric_comparison(summary, out_dir)
    plot_physical_layout(data, rows, args, out_dir)

    # Residual-risk surfaces for final policies.
    dqn_env = make_dqn_final_env(intensity, args, model, seed=args.eval_seed + 777)
    plot_residual_surface(dqn_env, "DQN sequential final residual risk", out_dir / "08_dqn_final_residual_risk_3d.png")

    paper_env = make_static_final_env(intensity, args, paper_row, seed=args.eval_seed + 778)
    plot_residual_surface(paper_env, "Paper-style static global greedy residual risk", out_dir / "09_paper_static_residual_risk_3d.png")

    region_env = make_static_final_env(intensity, args, region_row, seed=args.eval_seed + 779)
    plot_residual_surface(region_env, "Region-aware static greedy residual risk", out_dir / "10_region_aware_static_residual_risk_3d.png")

    # Write a compact report.
    report = [
        "# DQN Sensor Deployment Evaluation",
        "",
        "## Interpretation note",
        "",
        "The static greedy baselines are not movement-constrained. Treat them as placement-quality upper bounds.",
        "The DQN should mainly be compared against random_sequential and one_step_greedy_sequential, because those methods face the same movement and step constraints.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `dqn_sensor_policy.zip`: trained Stable-Baselines3 DQN model",
        "- `evaluation_summary.csv`: aggregate method comparison",
        "- `evaluation_episode_detail.csv`: per-episode results",
        "- `training_episode_rewards.csv`: training curve data",
        "- PNG plots: training curve, metric bars, physical layout, residual-risk surfaces",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))

    print("\nEvaluation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()