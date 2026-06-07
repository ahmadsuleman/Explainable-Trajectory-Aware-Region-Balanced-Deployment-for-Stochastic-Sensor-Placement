"""
explanatory_environment_plots_from_main_env.py

Companion plotting script for standard_trajectory_rl_environment.py.

This script does not redefine the main environment, trajectory generator,
intensity estimator, reward, or sensor model. It imports the main code and adds
explanatory annotations to the requested figures:

1. Generated physical trajectories in the 10 km x 10 km arena.
2. Trajectories mapped to representation/state space l=(alpha, p).
3. Counterfactual one-sensor reward distribution over the arena.
4. Sensor sensing model gamma(l,s) in representation space.

Place this file beside:
    standard_trajectory_rl_environment.py

Run:
    python explanatory_environment_plots_from_main_env.py
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from DQN_compatible_env import (
    TrajectoryGenerator,
    MAPLGCPEstimator,
    TrajectorySensorDeploymentEnv,
)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 450,
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_models(
    n_trajectories: int = 350,
    traffic_type: str = "multi_corridor",
    arena_size: float = 10.0,
    seed: int = 42,
):
    generator = TrajectoryGenerator(arena_size=arena_size, seed=seed)
    data = generator.generate(n=n_trajectories, traffic_type=traffic_type)

    estimator = MAPLGCPEstimator(
        n_alpha=30,
        n_p=30,
        p_min=-arena_size / 2.0,
        p_max=arena_size / 2.0,
        tau=2.0,
        kappa=1.2,
    )
    intensity = estimator.fit(data.rep_points)

    env = TrajectorySensorDeploymentEnv(
        intensity=intensity,
        arena_size=arena_size,
        expected_traffic=10.0,
        max_sensors=4,
        max_steps=30,
        motion_step=1.0,
        rho=0.95,
        sigma=0.45,
        min_sensor_spacing=0.5,
        seed=7,
    )
    return data, intensity, env


def plot_physical_trajectories_explained(data, arena_size: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 8.2))

    for p1, p2 in data.segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.85, alpha=0.28)

    # Pick one visible trajectory and annotate it as one sample path.
    sample_idx = min(20, len(data.segments) - 1)
    p1, p2 = data.segments[sample_idx]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2.6, alpha=0.95, label="one generated trajectory")
    mid = 0.5 * (p1 + p2)


    # Show arena boundary and center origin used for representation p.
    ax.add_patch(Rectangle((0, 0), arena_size, arena_size, fill=False, linewidth=1.8))
    ax.scatter([arena_size / 2], [arena_size / 2], s=55, marker="x", label="center used for p")


    ax.set_xlim(-0.15, arena_size + 0.15)
    ax.set_ylim(-0.15, arena_size + 0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position in environment (km)")
    ax.set_ylabel("y position in environment (km)")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_representation_state_space_explained(data, intensity, path: Path) -> None:
    alpha_deg = np.rad2deg(data.rep_points[:, 0])
    p_vals = data.rep_points[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6.9))
    extent = [
        np.rad2deg(intensity.alpha_edges[0]),
        np.rad2deg(intensity.alpha_edges[-1]),
        intensity.p_edges[0],
        intensity.p_edges[-1],
    ]
    img = ax.imshow(
        intensity.lambda_plot.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
        alpha=0.82,
        cmap="viridis",
    )
    ax.scatter(alpha_deg, p_vals, s=13, alpha=0.52, edgecolors="none", label="mapped trajectories")

    # Mark the same trajectory from plot 1 in representation space.
    sample_idx = min(20, len(data.rep_points) - 1)
    a0 = float(np.rad2deg(data.rep_points[sample_idx, 0]))
    p0 = float(data.rep_points[sample_idx, 1])
    ax.scatter([a0], [p0], s=95, marker="o", edgecolors="black", linewidths=1.2, label="one path as one state point")
    


   
    ax.set_xlim(0, 180)
    ax.set_ylim(intensity.p_edges[0], intensity.p_edges[-1])
    ax.set_xlabel(r"normal angle $\alpha$ (degrees)")
    ax.set_ylabel(r"perpendicular offset $p$ from center (km)")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_reward_distribution_explained(env: TrajectorySensorDeploymentEnv, path: Path, grid_n: int = 51) -> None:
    env.reset(start_xy=[env.arena_size / 2.0, env.arena_size / 2.0])
    xs, ys, surfaces = env.counterfactual_placement_surface(nx=grid_n, ny=grid_n)
    Z = surfaces["reward"]
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(10.6, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    valid = np.isfinite(Z)
    norm = Normalize(vmin=float(np.nanmin(Z[valid])), vmax=float(np.nanmax(Z[valid])))
    facecolors = cm.viridis(norm(Z))
    ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(Z)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("immediate one-sensor reward")

    # Annotate the highest reward location without drawing a sensor marker.


    ax.set_xlabel("x position in environment (km)")
    ax.set_ylabel("y position in environment (km)")
    ax.set_zlabel("reward")
    ax.view_init(elev=31, azim=-133)
    save_figure(fig, path)


def plot_sensor_sensing_model_explained(env: TrajectorySensorDeploymentEnv, path: Path, sensor_xy=None) -> None:
    if sensor_xy is None:
        sensor_xy = np.array([env.arena_size / 2.0, env.arena_size / 2.0], dtype=float)
    else:
        sensor_xy = np.asarray(sensor_xy, dtype=float)

    Gamma = env.sensor_detection_map(sensor_xy)
    A_deg, P = np.meshgrid(np.rad2deg(env.alpha_centers), env.p_centers, indexing="ij")

    fig = plt.figure(figsize=(10.6, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(np.min(Gamma)), vmax=float(np.max(Gamma)))
    facecolors = cm.plasma(norm(Gamma))
    ax.plot_surface(A_deg, P, Gamma, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)

    mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
    mappable.set_array(Gamma)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("sensor response")


    ax.set_xlabel(r"normal angle $\alpha$ (degrees)")
    ax.set_ylabel(r"offset $p$ (km)")
    ax.set_zlabel(r"$\gamma(l,s)$")
    ax.view_init(elev=30, azim=-130)
    save_figure(fig, path)


def main(output_dir: str = "explanatory_environment_plot_outputs") -> None:
    configure_matplotlib()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data, intensity, env = build_models()

    files = [
        out / "01_explained_physical_trajectory_generation.png",
        out / "02_explained_representation_state_space.png",
        out / "03_explained_single_sensor_reward_distribution_3d.png",
        out / "04_explained_sensor_sensing_model_3d.png",
    ]

    plot_physical_trajectories_explained(data, env.arena_size, files[0])
    plot_representation_state_space_explained(data, intensity, files[1])
    plot_reward_distribution_explained(env, files[2], grid_n=51)
    plot_sensor_sensing_model_explained(env, files[3], sensor_xy=[env.arena_size / 2.0, env.arena_size / 2.0])

    # readme = out / "README.txt"
    # readme.write_text(
    #     "Explanatory environment plots generated from the main environment module.\n\n"
    #     "The script imports standard_trajectory_rl_environment.py and adds explanatory annotations only.\n"
    #     "It does not redefine the environment mechanics and does not show final sensor deployment results.\n\n"
    #     "Figures:\n"
    #     "1. Physical trajectory generation with an annotated sample trajectory.\n"
    #     "2. Mapping from physical trajectory to representation/state point l=(alpha,p).\n"
    #     "3. Counterfactual one-sensor reward field over the physical arena.\n"
    #     "4. Sensor sensing response gamma(l,s) in representation space.\n",
    #     encoding="utf-8",
    # )

    # zip_path = Path("explanatory_environment_plot_outputs.zip")
    # with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    #     for p in files + [readme]:
    #         zf.write(p, arcname=p.name)

    print(f"Saved explanatory figures to: {out}")
    # print(f"Saved ZIP to: {zip_path}")


if __name__ == "__main__":
    main()