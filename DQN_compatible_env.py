"""
standard_trajectory_rl_environment.py

Final clean environment for trajectory-aware sensor deployment.

What this file provides
-----------------------
1. Synthetic stochastic trajectory generation in a 10 km x 10 km arena.
2. Mapping of physical trajectories into representation space (alpha, p).
3. MAP-smoothed Poisson log-intensity approximation:
      trajectory points -> smoothed intensity lambda_hat(alpha, p)
4. A complete RL-style environment:
      continuous state, discrete actions, model-based motion, sensor placement
5. Standard trajectory-aware metrics:
      expected undetected mass U(S)
      void probability V(S)=exp(-U(S))
      log-void value log V(S)=-U(S)
      detection score D(S)
      residual-risk map R(l,S)=lambda_hat(l)*pi(l,S)
6. Explainable balanced reward:
      immediate reward returns KPI breakdown for every action
      includes total risk reduction and worst-trajectory-region coverage terms
7. Baseline hooks:
      random agent and greedy immediate-KPI agent
8. Demo plots and CSV exports when run as a script.

Author note
-----------
This is not full R-INLA. It is a MAP-smoothed Poisson log-intensity
estimator: Poisson counts + a smooth Gaussian prior on the log-intensity field.
The code is designed as a toy simulator with internally consistent mechanisms.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:  # Optional: used only to expose Gymnasium spaces when installed.
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

try:
    from scipy.optimize import minimize
    from scipy.sparse import eye, lil_matrix
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This environment needs scipy for the MAP-LGCP estimator. "
        "Install scipy or replace MAPLGCPEstimator with a histogram/KDE estimator."
    ) from exc


# =============================================================================
# 1. Trajectory generation and representation-space mapping
# =============================================================================

@dataclass
class TrajectoryData:
    segments: List[Tuple[np.ndarray, np.ndarray]]
    rep_points: np.ndarray  # shape: (n,2), columns: alpha, p


class TrajectoryGenerator:
    """
    Generates stochastic linear trajectories and maps them into representation space.

    Representation:
        l = (alpha, p) uses the normal-form line equation
            x*cos(alpha) + y*sin(alpha) = p
        alpha in [0, pi) is the line-normal angle.
        p is the signed perpendicular offset in centered arena coordinates.

    This avoids slope singularities and makes distance-to-line calculations exact.
    """

    def __init__(self, arena_size: float = 10.0, seed: int = 42):
        self.arena_size = float(arena_size)
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int = 300, traffic_type: str = "multi_corridor") -> TrajectoryData:
        alphas, ps = self._sample_representation_points(n, traffic_type)
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        rep_points: List[Tuple[float, float]] = []

        for alpha, p in zip(alphas, ps):
            seg = self.clip_line_to_square(alpha, p)
            if seg is None:
                continue
            p1, p2 = seg
            alpha_hat, p_hat = self.segment_to_representation(p1, p2)
            segments.append((p1, p2))
            rep_points.append((alpha_hat, p_hat))

        return TrajectoryData(segments=segments, rep_points=np.asarray(rep_points, dtype=float))

    def _sample_representation_points(self, n: int, traffic_type: str):
        alphas = []
        ps = []

        for _ in range(n):
            if traffic_type == "single_corridor":
                alpha = self.rng.normal(np.deg2rad(55), np.deg2rad(6))
                p = self.rng.normal(0.0, 0.8)

            elif traffic_type == "multi_corridor":
                component = self.rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])
                alpha_means = [np.deg2rad(45), np.deg2rad(90), np.deg2rad(130)]
                p_means = [-1.2, 0.5, 1.6]
                alpha = self.rng.normal(alpha_means[component], np.deg2rad(6))
                p = self.rng.normal(p_means[component], 0.6)

            elif traffic_type == "dispersed":
                alpha = self.rng.uniform(0, np.pi)
                p = self.rng.uniform(-4.5, 4.5)

            else:
                raise ValueError(f"Unknown traffic_type: {traffic_type}")

            alphas.append(alpha % np.pi)
            ps.append(p)

        return np.asarray(alphas), np.asarray(ps)

    def clip_line_to_square(self, alpha: float, p: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert the normal-form representation-space line into a physical segment
        clipped to [0, arena_size]^2.

        Line equation in centered coordinates:
            x*cos(alpha) + y*sin(alpha) = p
        """
        half = self.arena_size / 2.0
        alpha = float(alpha % np.pi)
        p = float(p)
        c = float(np.cos(alpha))
        s = float(np.sin(alpha))
        eps = 1e-10
        points = []

        # Intersections with vertical square boundaries x = +/- half.
        if abs(s) > eps:
            for x in (-half, half):
                y = (p - x * c) / s
                if -half - 1e-9 <= y <= half + 1e-9:
                    points.append((x, float(np.clip(y, -half, half))))

        # Intersections with horizontal square boundaries y = +/- half.
        if abs(c) > eps:
            for y in (-half, half):
                x = (p - y * s) / c
                if -half - 1e-9 <= x <= half + 1e-9:
                    points.append((float(np.clip(x, -half, half)), y))

        unique = []
        for pt in points:
            if not any(np.linalg.norm(np.asarray(pt) - np.asarray(q)) < 1e-6 for q in unique):
                unique.append(pt)

        if len(unique) < 2:
            return None

        best_pair = None
        best_dist = -np.inf
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                d = np.linalg.norm(np.asarray(unique[i]) - np.asarray(unique[j]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (unique[i], unique[j])

        p1, p2 = best_pair
        return np.asarray(p1, dtype=float) + half, np.asarray(p2, dtype=float) + half

    def segment_to_representation(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
        """Map a physical segment back to the normal-form representation."""
        half = self.arena_size / 2.0
        q1 = np.asarray(p1, dtype=float) - half
        q2 = np.asarray(p2, dtype=float) - half
        d = q2 - q1
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            raise ValueError("Cannot represent a zero-length segment.")

        # A unit normal to the segment direction.
        n = np.array([-d[1], d[0]], dtype=float) / length
        alpha = float(np.arctan2(n[1], n[0]))
        p = float(n @ q1)

        # Canonicalize to alpha in [0, pi). If we flip the normal, p must flip too.
        if alpha < 0.0:
            alpha += np.pi
            p = -p
        elif alpha >= np.pi:
            alpha -= np.pi
            p = -p

        if np.isclose(alpha, np.pi, atol=1e-12):
            alpha = 0.0
            p = -p

        return alpha, p


# =============================================================================
# 2. MAP-smoothed Poisson log-intensity approximation
# =============================================================================

@dataclass
class IntensityEstimate:
    lambda_grid: np.ndarray
    lambda_plot: np.ndarray
    counts: np.ndarray
    alpha_edges: np.ndarray
    p_edges: np.ndarray
    alpha_centers: np.ndarray
    p_centers: np.ndarray
    success: bool
    message: str


class MAPLGCPEstimator:
    """
    Discretized MAP-smoothed Poisson log-intensity estimator.

    Observation model:
        y_j ~ Poisson(A_j * exp(eta_j))

    Prior:
        eta ~ N(0, Q^-1)

    MAP objective:
        L(eta) = sum_j [A_j exp(eta_j) - y_j eta_j] + 0.5 eta^T Q eta

    This is a lightweight MAP estimator, not full INLA. It intentionally does not
    compute posterior marginals, hyperparameter posteriors, or uncertainty bands.
    """

    def __init__(
        self,
        n_alpha: int = 30,
        n_p: int = 30,
        p_min: float = -5.0,
        p_max: float = 5.0,
        tau: float = 2.0,
        kappa: float = 1.2,
    ):
        self.n_alpha = int(n_alpha)
        self.n_p = int(n_p)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.tau = float(tau)
        self.kappa = float(kappa)

    def fit(self, rep_points: np.ndarray) -> IntensityEstimate:
        alpha_vals = rep_points[:, 0]
        p_vals = rep_points[:, 1]

        counts, alpha_edges, p_edges = np.histogram2d(
            alpha_vals,
            p_vals,
            bins=[self.n_alpha, self.n_p],
            range=[[0, np.pi], [self.p_min, self.p_max]],
            density=False,
        )

        y = counts.reshape(-1)
        d_alpha = alpha_edges[1] - alpha_edges[0]
        d_p = p_edges[1] - p_edges[0]
        cell_area = d_alpha * d_p

        Q = self._build_precision_matrix(d_alpha=d_alpha, d_p=d_p)

        eta0 = np.log(counts.reshape(-1) + 0.1)

        def objective(eta):
            rate = cell_area * np.exp(eta)
            poisson_nll = np.sum(rate - y * eta)
            prior = 0.5 * eta @ (Q @ eta)
            return poisson_nll + prior

        def gradient(eta):
            rate = cell_area * np.exp(eta)
            return (rate - y) + (Q @ eta)

        result = minimize(
            objective,
            eta0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-8},
        )

        eta_hat = result.x.reshape(self.n_alpha, self.n_p)
        lambda_grid = np.exp(eta_hat)
        lambda_plot = lambda_grid / max(float(np.max(lambda_grid)), 1e-12)

        alpha_centers = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])
        p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])

        return IntensityEstimate(
            lambda_grid=lambda_grid,
            lambda_plot=lambda_plot,
            counts=counts,
            alpha_edges=alpha_edges,
            p_edges=p_edges,
            alpha_centers=alpha_centers,
            p_centers=p_centers,
            success=bool(result.success),
            message=str(result.message),
        )

    def _build_precision_matrix(self, d_alpha: float, d_p: float):
        """
        Build a grid precision matrix with:
        - periodic smoothing in alpha because alpha lives on [0, pi),
        - non-periodic smoothing in p,
        - grid-spacing-aware penalties so changing resolution does not silently
          change the meaning of the prior.
        """
        n = self.n_alpha * self.n_p
        L = lil_matrix((n, n))
        w_alpha = 1.0 / max(float(d_alpha) ** 2, 1e-12)
        w_p = 1.0 / max(float(d_p) ** 2, 1e-12)

        def idx(i, j):
            return i * self.n_p + j

        for i in range(self.n_alpha):
            for j in range(self.n_p):
                current = idx(i, j)
                neighbors = []

                if self.n_alpha > 1:
                    neighbors.append((idx((i - 1) % self.n_alpha, j), w_alpha))
                    neighbors.append((idx((i + 1) % self.n_alpha, j), w_alpha))

                if j > 0:
                    neighbors.append((idx(i, j - 1), w_p))
                if j < self.n_p - 1:
                    neighbors.append((idx(i, j + 1), w_p))

                # Avoid duplicate neighbor entries when n_alpha == 2.
                merged = {}
                for nb, weight in neighbors:
                    if nb == current:
                        continue
                    merged[nb] = merged.get(nb, 0.0) + weight

                for nb, weight in merged.items():
                    L[current, current] += weight
                    L[current, nb] -= weight

        return self.tau * (eye(n, format="csr") + self.kappa * L.tocsr())


# =============================================================================
# 3. RL Environment
# =============================================================================

class TrajectorySensorDeploymentEnv:
    """
    Standard RL-style environment.

    State:
        continuous vector + residual-risk map + placed sensors.

    Actions:
        0 move+north
        1 move+south
        2 move+east
        3 move+west
        4 move+stay
        5 place

    Metrics:
        U(S) = sum_l lambda_hat(l) * pi(l,S)
        V(S) = exp(-U(S))
        logV(S) = -U(S)
        D(S) = 1 - U(S)/sum_l lambda_hat(l)
        R(l,S) = lambda_hat(l) * pi(l,S)
  U_k(S) = regional residual risk over coarse trajectory-region k
  D_min(S) = worst-region detection score

    Reward:
        immediate KPI improvement + movement/invalid penalties + optional terminal score.
    """

    ACTIONS = [
        ("move", "north"),
        ("move", "south"),
        ("move", "east"),
        ("move", "west"),
        ("place", "stay"),
    ]

    def __init__(
        self,
        intensity: IntensityEstimate,
        arena_size: float = 10.0,
        expected_traffic: float = 10.0,
        max_sensors: int = 4,
        max_steps: int = 30,
        motion_step: float = 1.0,
        rho: float = 0.95,
        sigma: float = 0.45,
        min_sensor_spacing: float = 0.5,
        redundancy_sigma: float = 1.5,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: int = 123,
    ):
        self.intensity = intensity
        self.arena_size = float(arena_size)
        self.expected_traffic = float(expected_traffic)
        self.max_sensors = int(max_sensors)
        self.max_steps = int(max_steps)
        self.motion_step = float(motion_step)
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.min_sensor_spacing = float(min_sensor_spacing)
        self.redundancy_sigma = float(redundancy_sigma)
        self.rng = np.random.default_rng(seed)

        if reward_weights is None:
            reward_weights = {
                # One primary objective: normalized reduction in expected undetected traffic.
                "risk_reduction_norm": 1.0,

                # Optional fairness term: improve the least-covered active trajectory region.
                "worst_detection_gain": 1.0,

                # Deployment and constraint costs.
                "step_cost": 0.001,
                "move_cost": 0.001,
                "invalid": 0.25,
                "placement_cost": 0.0,
                "redundancy": 0.02,

                # Terminal objective terms. These are deliberately stronger than step costs
                # so a greedy or RL policy is not rewarded for wasting the episode.
                "terminal_detection": 1.0,
                "terminal_worst_detection": 1.05,
                "terminal_unplaced_penalty": 1.00,
            }
        self.reward_weights = reward_weights

        self.alpha_centers = intensity.alpha_centers
        self.p_centers = intensity.p_centers
        self.alpha_grid, self.p_grid = np.meshgrid(self.alpha_centers, self.p_centers, indexing="ij")
        self.shape = intensity.lambda_grid.shape

        lam = np.maximum(intensity.lambda_grid, 0.0)
        self.traffic_weights = self.expected_traffic * lam / max(float(np.sum(lam)), 1e-12)
        self.total_expected = float(np.sum(self.traffic_weights))
        self.initial_risk = self.total_expected

        if spaces is not None:
            self.action_space = spaces.Discrete(len(self.ACTIONS))
            self.observation_space = spaces.Dict(
                {
                    "features": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
                    "residual_map": spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32),
                    "sensor_locations": spaces.Box(low=0.0, high=1.0, shape=(self.max_sensors, 2), dtype=np.float32),
                    "agent_xy": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        self.reset()

    @property
    def n_actions(self):
        return len(self.ACTIONS)

    def reset(self, start_xy: Optional[List[float]] = None, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start_xy is None:
            self.agent_xy = np.array(
                [self.rng.uniform(0, self.arena_size), self.rng.uniform(0, self.arena_size)],
                dtype=float,
            )
        else:
            self.agent_xy = np.asarray(start_xy, dtype=float)

        self.sensors: List[np.ndarray] = []
        self.step_count = 0
        self.done = False
        self.current_metrics = self.compute_metrics(self.sensors)
        return self.get_observation()

    def step(self, action_id: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        action_id = int(action_id)
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError(f"action_id must be in [0, {self.n_actions - 1}], got {action_id}.")

        action_type, motion = self.ACTIONS[action_id]
        old_metrics = self.current_metrics
        old_xy = self.agent_xy.copy()
        old_sensors = [s.copy() for s in self.sensors]

        invalid = False
        placed = False
        redundancy_penalty = 0.0
        move_distance = 0.0

        if action_type == "place":
            if self.can_place_sensor(self.agent_xy):
                redundancy_penalty = self.compute_redundancy_penalty(self.agent_xy, old_sensors)
                self.sensors.append(self.agent_xy.copy())
                placed = True
            else:
                invalid = True
        elif action_type == "move":
            self.agent_xy = self.move(self.agent_xy, motion)
            move_distance = float(np.linalg.norm(self.agent_xy - old_xy))
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        self.step_count += 1
        new_metrics = self.compute_metrics(self.sensors)
        self.current_metrics = new_metrics

        terminated = len(self.sensors) >= self.max_sensors
        truncated = self.step_count >= self.max_steps
        self.done = terminated or truncated

        reward, breakdown = self.compute_reward(
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            move_distance=move_distance,
            invalid=invalid,
            placed=placed,
            terminal=self.done,
            redundancy_penalty=redundancy_penalty,
        )

        info = {
            "action_id": action_id,
            "action": self.ACTIONS[action_id],
            "placed": placed,
            "invalid": invalid,
            "num_sensors": len(self.sensors),
            "sensors": np.asarray(self.sensors, dtype=float),
            "metrics": new_metrics,
            "reward_breakdown": breakdown,
            "terminated": terminated,
            "truncated": truncated,
        }

        return self.get_observation(), reward, self.done, info

    def step_gym(self, action_id: int):
        """Gymnasium-style step: obs, reward, terminated, truncated, info."""
        obs, reward, _, info = self.step(action_id)
        return obs, reward, bool(info["terminated"]), bool(info["truncated"]), info

    # -------------------------------------------------------------------------
    # Motion and constraints
    # -------------------------------------------------------------------------

    def move(self, xy: np.ndarray, motion: str) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        h = self.motion_step

        if motion == "north":
            y += h
        elif motion == "south":
            y -= h
        elif motion == "east":
            x += h
        elif motion == "west":
            x -= h
        elif motion == "stay":
            pass
        else:
            raise ValueError(f"Unknown motion: {motion}")

        return np.array([np.clip(x, 0, self.arena_size), np.clip(y, 0, self.arena_size)], dtype=float)

    def can_place_sensor(self, xy: np.ndarray) -> bool:
        if len(self.sensors) >= self.max_sensors:
            return False
        if len(self.sensors) == 0:
            return True
        d = np.linalg.norm(np.asarray(self.sensors) - xy[None, :], axis=1)
        return float(np.min(d)) >= self.min_sensor_spacing

    # -------------------------------------------------------------------------
    # Detection, thinning, residual risk, metrics
    # -------------------------------------------------------------------------

    def line_distance_to_sensor(self, alpha: np.ndarray, p: np.ndarray, sensor_xy: np.ndarray):
        """
        Exact perpendicular distance from a sensor to each normal-form line:
            x*cos(alpha) + y*sin(alpha) = p
        using centered arena coordinates.
        """
        half = self.arena_size / 2.0
        x = float(sensor_xy[0]) - half
        y = float(sensor_xy[1]) - half
        return np.abs(x * np.cos(alpha) + y * np.sin(alpha) - p)

    def sensor_detection_map(self, sensor_xy: np.ndarray):
        d = self.line_distance_to_sensor(self.alpha_grid, self.p_grid, sensor_xy)
        gamma = self.rho * np.exp(-(d**2) / (2.0 * max(self.sigma, 1e-12) ** 2))
        return np.clip(gamma, 0.0, 1.0)

    def miss_probability_map(self, sensors: List[np.ndarray]):
        miss = np.ones(self.shape, dtype=float)
        for sensor in sensors:
            gamma = self.sensor_detection_map(sensor)
            miss *= 1.0 - gamma
        return miss

    def compute_metrics(self, sensors: List[np.ndarray]):
        miss_map = self.miss_probability_map(sensors)
        residual_map = self.traffic_weights * miss_map

        U = float(np.sum(residual_map))
        V = float(np.exp(-U))
        logV = float(-U)
        D = float(1.0 - U / max(self.total_expected, 1e-12))
        D = float(np.clip(D, 0.0, 1.0))

        max_residual = float(np.max(residual_map))
        risk_flat = residual_map.reshape(-1)
        risk_sum = float(np.sum(risk_flat))


        if risk_sum > 0:
            probs = risk_flat / risk_sum
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropy = entropy / np.log(len(probs))
            idx = int(np.argmax(residual_map))
            i, j = np.unravel_index(idx, residual_map.shape)
            dominant_alpha = float(self.alpha_grid[i, j])
            dominant_p = float(self.p_grid[i, j])
        else:
            entropy = 0.0
            dominant_alpha = 0.0
            dominant_p = 0.0

        regional = self.regional_risk_metrics(residual_map)

        return {
            "expected_undetected": U,
            "void_probability": V,
            "log_void": logV,
            "detection_score": D,
            "max_residual_risk": max_residual,
            "risk_entropy": entropy,
            "dominant_alpha": dominant_alpha,
            "dominant_p": dominant_p,
            "miss_map": miss_map,
            "residual_map": residual_map,
            **regional,
        }

    # -------------------------------------------------------------------------
    # Reward and observation
    # -------------------------------------------------------------------------

    def compute_redundancy_penalty(self, candidate_xy: np.ndarray, existing_sensors: List[np.ndarray]) -> float:
        """
        Penalizes placing a new sensor very close to existing sensors.

        This prevents the reward from repeatedly selecting clustered placements
        around the strongest trajectory mode unless the KPI improvement is large enough
        to justify that redundancy.

        P_red = exp(-d_min^2 / redundancy_sigma)
        """
        if len(existing_sensors) == 0:
            return 0.0
        d = np.linalg.norm(np.asarray(existing_sensors) - candidate_xy[None, :], axis=1)
        d_min = float(np.min(d))
        return float(np.exp(-(d_min**2) / max(self.redundancy_sigma, 1e-12)))

    def compute_reward(
        self,
        old_metrics: Dict,
        new_metrics: Dict,
        move_distance: float,
        invalid: bool,
        placed: bool,
        terminal: bool,
        redundancy_penalty: float = 0.0,
    ):
        """
        Clean toy reward:
            normalized global risk reduction
          + optional worst-active-region detection improvement
          - simple deployment costs.

        This avoids double-counting log-void, detection gain, and risk reduction,
        which are all transformations of the same expected-undetected mass U(S).
        """
        w = self.reward_weights

        risk_reduction = old_metrics["expected_undetected"] - new_metrics["expected_undetected"]
        risk_reduction_norm = risk_reduction / max(self.initial_risk, 1e-12)
        worst_detection_gain = (
            new_metrics["worst_region_detection"] - old_metrics["worst_region_detection"]
        )

        invalid_cost = 1.0 if invalid else 0.0
        placement_cost = 1.0 if placed else 0.0

        immediate = (
            10 * risk_reduction_norm
            + 10 * worst_detection_gain
            # - w["step_cost"]
            # - w["move_cost"] * move_distance
            - 2 * invalid_cost
            # - w["placement_cost"] * placement_cost
            - 2 * redundancy_penalty
        )

        terminal_reward = 0.0
        if terminal:
            unplaced = max(0, self.max_sensors - len(self.sensors))
            terminal_reward = (
                w["terminal_detection"] * new_metrics["detection_score"]
                + w["terminal_worst_detection"] * new_metrics["worst_region_detection"]
                - w["terminal_unplaced_penalty"] * (unplaced / max(self.max_sensors, 1))
            )

        total = immediate + terminal_reward

        return float(total), {
            "risk_reduction": float(risk_reduction),
            "risk_reduction_norm": float(risk_reduction_norm),
            "worst_detection_gain": float(worst_detection_gain),
            "step_cost": float(w["step_cost"]),
            "move_cost": float(move_distance),
            "invalid_cost": float(invalid_cost),
            "placement_cost": float(placement_cost),
            "redundancy_penalty": float(redundancy_penalty),
            "terminal_reward": float(terminal_reward),
            "total_reward": float(total),
        }

    def get_observation(self):
        m = len(self.sensors)
        metrics = self.current_metrics
        p_min = self.intensity.p_edges[0]
        p_max = self.intensity.p_edges[-1]

        residual = metrics["residual_map"]
        residual_norm = residual / max(float(np.max(residual)), 1e-12)

        sensor_pad = np.zeros((self.max_sensors, 2), dtype=np.float32)
        for i, s in enumerate(self.sensors[: self.max_sensors]):
            sensor_pad[i, :] = s / self.arena_size

        features = np.array(
            [
                self.agent_xy[0] / self.arena_size,
                self.agent_xy[1] / self.arena_size,
                m / self.max_sensors,
                (self.max_sensors - m) / self.max_sensors,
                self.step_count / self.max_steps,
                metrics["void_probability"],
                metrics["log_void"] / max(self.expected_traffic, 1e-12),
                metrics["detection_score"],
                metrics["expected_undetected"] / max(self.expected_traffic, 1e-12),
                metrics["max_residual_risk"] / max(self.expected_traffic, 1e-12),
                metrics["risk_entropy"],
                metrics["worst_region_detection"],
                metrics["worst_region_U"] / max(self.expected_traffic, 1e-12),
                metrics["mean_region_detection"],
                metrics["dominant_alpha"] / np.pi,
                (metrics["dominant_p"] - p_min) / (p_max - p_min),
            ],
            dtype=np.float32,
        )

        return {
            "features": features,
            "residual_map": residual_norm.astype(np.float32),
            "sensor_locations": sensor_pad,
            "agent_xy": (self.agent_xy / self.arena_size).astype(np.float32),
        }

    # -------------------------------------------------------------------------
    # Explainability and baselines
    # -------------------------------------------------------------------------

    def trajectory_risk_table(self) -> List[Dict]:
        """Expose intensity, miss probability, and residual risk at every representation-space point."""
        rows = []
        metrics = self.current_metrics
        miss = metrics["miss_map"]
        residual = metrics["residual_map"]

        for i, alpha in enumerate(self.alpha_centers):
            for j, p in enumerate(self.p_centers):
                rows.append(
                    {
                        "alpha_rad": float(alpha),
                        "alpha_deg": float(np.rad2deg(alpha)),
                        "p_km": float(p),
                        "lambda_weight": float(self.traffic_weights[i, j]),
                        "miss_probability": float(miss[i, j]),
                        "residual_risk": float(residual[i, j]),
                    }
                )
        return rows

    def evaluate_action_consequences(self) -> List[Dict]:
        """Evaluate all 10 discrete actions without permanently changing environment state."""
        saved_xy = self.agent_xy.copy()
        saved_sensors = [s.copy() for s in self.sensors]
        saved_step = self.step_count
        saved_done = self.done
        saved_metrics = self.current_metrics

        rows = []

        for action_id in range(self.n_actions):
            self.agent_xy = saved_xy.copy()
            self.sensors = [s.copy() for s in saved_sensors]
            self.step_count = saved_step
            self.done = False
            self.current_metrics = saved_metrics

            _, reward, _, info = self.step(action_id)
            row = {
                "action_id": action_id,
                "action": f"{info['action'][0]}+{info['action'][1]}",
                "placed": info["placed"],
                "invalid": info["invalid"],
                "reward": reward,
            }
            row.update(info["reward_breakdown"])
            rows.append(row)

        self.agent_xy = saved_xy
        self.sensors = saved_sensors
        self.step_count = saved_step
        self.done = saved_done
        self.current_metrics = saved_metrics

        return rows

    def regional_risk_metrics(self, residual_map, n_alpha_regions=5, n_p_regions=5, active_fraction=0.01):
        """
        Computes residual-risk metrics over coarse representation-space regions.
        Worst-region detection is computed only over active regions so nearly empty
        bins do not dominate the fairness term numerically.
        """
        alpha_splits = np.array_split(np.arange(residual_map.shape[0]), n_alpha_regions)
        p_splits = np.array_split(np.arange(residual_map.shape[1]), n_p_regions)

        region_U = []
        region_total = []

        for a_idx in alpha_splits:
            for p_idx in p_splits:
                block_residual = residual_map[np.ix_(a_idx, p_idx)]
                block_total = self.traffic_weights[np.ix_(a_idx, p_idx)]

                region_U.append(np.sum(block_residual))
                region_total.append(np.sum(block_total))

        region_U = np.array(region_U, dtype=float)
        region_total = np.array(region_total, dtype=float)
        region_detection = 1.0 - region_U / np.maximum(region_total, 1e-12)

        active = region_total > active_fraction * max(float(np.sum(region_total)), 1e-12)
        if not np.any(active):
            active = region_total > 0.0
        if not np.any(active):
            active = np.ones_like(region_total, dtype=bool)

        return {
            "region_U": region_U,
            "region_detection": region_detection,
            "active_region_mask": active,
            "worst_region_U": float(np.max(region_U[active])),
            "worst_region_detection": float(np.min(region_detection[active])),
            "mean_region_detection": float(np.mean(region_detection[active])),
        }

    def counterfactual_placement_surface(self, nx: int = 31, ny: int = 31):
        """
        For explainability:
        "If a sensor were placed at every physical location, what immediate KPI reward would result?"
        """
        xs = np.linspace(0, self.arena_size, nx)
        ys = np.linspace(0, self.arena_size, ny)

        reward = np.zeros((ny, nx))
        log_void_gain = np.zeros((ny, nx))
        detection_gain = np.zeros((ny, nx))
        risk_reduction = np.zeros((ny, nx))
        worst_detection_gain = np.zeros((ny, nx))
        worst_risk_reduction = np.zeros((ny, nx))
        redundancy_penalty_surface = np.zeros((ny, nx))

        old_metrics = self.current_metrics

        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                candidate = np.array([x, y], dtype=float)
                if not self.can_place_sensor(candidate):
                    reward[iy, ix] = np.nan
                    log_void_gain[iy, ix] = np.nan
                    detection_gain[iy, ix] = np.nan
                    risk_reduction[iy, ix] = np.nan
                    worst_detection_gain[iy, ix] = np.nan
                    worst_risk_reduction[iy, ix] = np.nan
                    redundancy_penalty_surface[iy, ix] = np.nan
                    continue

                sensors2 = [s.copy() for s in self.sensors] + [candidate]
                new_metrics = self.compute_metrics(sensors2)
                red_penalty = self.compute_redundancy_penalty(candidate, self.sensors)
                r, br = self.compute_reward(
                    old_metrics,
                    new_metrics,
                    move_distance=0.0,
                    invalid=False,
                    placed=True,
                    terminal=False,
                    redundancy_penalty=red_penalty,
                )
                reward[iy, ix] = r
                log_void_gain[iy, ix] = new_metrics["log_void"] - old_metrics["log_void"]
                detection_gain[iy, ix] = new_metrics["detection_score"] - old_metrics["detection_score"]
                risk_reduction[iy, ix] = br["risk_reduction"]
                worst_detection_gain[iy, ix] = br["worst_detection_gain"]
                worst_risk_reduction[iy, ix] = old_metrics["worst_region_U"] - new_metrics["worst_region_U"]
                redundancy_penalty_surface[iy, ix] = br["redundancy_penalty"]

        return xs, ys, {
            "reward": reward,
            "log_void_gain": log_void_gain,
            "detection_gain": detection_gain,
            "risk_reduction": risk_reduction,
            "worst_detection_gain": worst_detection_gain,
            "worst_risk_reduction": worst_risk_reduction,
            "redundancy_penalty": redundancy_penalty_surface,
        }

    def validate_mechanism(self, atol: float = 1e-9) -> Dict[str, bool]:
        """
        Run invariant checks that define a correct toy mechanism.
        Raises AssertionError if a core mathematical property is broken.
        """
        empty = self.compute_metrics([])
        assert np.all(empty["miss_map"] >= -atol)
        assert np.all(empty["miss_map"] <= 1.0 + atol)
        assert np.allclose(empty["miss_map"], 1.0, atol=atol)
        assert abs(empty["expected_undetected"] - self.total_expected) <= max(atol, 1e-8)
        assert 0.0 - atol <= empty["detection_score"] <= 1.0 + atol

        center = np.array([self.arena_size / 2.0, self.arena_size / 2.0], dtype=float)
        alpha0 = np.array([[0.0]])
        p0 = np.array([[0.0]])
        d0 = self.line_distance_to_sensor(alpha0, p0, center)
        gamma0 = self.rho * np.exp(-(d0**2) / (2.0 * max(self.sigma, 1e-12) ** 2))
        assert np.isclose(float(gamma0[0, 0]), self.rho, atol=1e-8)

        if self.max_sensors > 0:
            one = self.compute_metrics([center])
            assert one["expected_undetected"] <= empty["expected_undetected"] + atol
            assert np.all(one["miss_map"] >= -atol)
            assert np.all(one["miss_map"] <= 1.0 + atol)
            assert 0.0 - atol <= one["detection_score"] <= 1.0 + atol

        return {
            "empty_miss_is_one": True,
            "risk_is_monotone_after_adding_sensor": True,
            "detection_score_is_bounded": True,
            "zero_distance_detection_equals_rho": True,
        }

    def run_random_episode(self):
        self.reset()
        total_reward = 0.0
        last_info = None
        while not self.done:
            a = int(self.rng.integers(0, self.n_actions))
            _, r, done, info = self.step(a)
            total_reward += r
            last_info = info
        return total_reward, last_info

    def run_greedy_immediate_episode(self, start_xy: Optional[List[float]] = None, grid_n: int = 31):
        """
        Greedy placement baseline for testing the sensor mechanism directly.

        This is intentionally not a path-planning policy. It searches over a physical
        grid, places the next sensor at the valid location with the largest immediate
        risk reduction, and repeats until max_sensors is reached. Keeping this baseline
        separate from movement avoids falsely blaming the sensing mechanism for a
        one-step movement-credit assignment problem.
        """
        self.reset(start_xy=start_xy)
        total_reward = 0.0
        history = []
        place_action_id = self.ACTIONS.index(("place", "stay"))
        xs = np.linspace(0.0, self.arena_size, int(grid_n))
        ys = np.linspace(0.0, self.arena_size, int(grid_n))

        while len(self.sensors) < self.max_sensors and self.step_count < self.max_steps:
            old_metrics = self.current_metrics
            best = None

            for x in xs:
                for y in ys:
                    candidate = np.array([x, y], dtype=float)
                    if not self.can_place_sensor(candidate):
                        continue
                    sensors2 = [s.copy() for s in self.sensors] + [candidate]
                    new_metrics = self.compute_metrics(sensors2)
                    risk_reduction = old_metrics["expected_undetected"] - new_metrics["expected_undetected"]
                    worst_gain = new_metrics["worst_region_detection"] - old_metrics["worst_region_detection"]
                    score = risk_reduction / max(self.initial_risk, 1e-12) + 0.25 * worst_gain
                    if best is None or score > best["score"]:
                        best = {
                            "xy": candidate,
                            "score": float(score),
                            "risk_reduction": float(risk_reduction),
                            "worst_detection_gain": float(worst_gain),
                        }

            if best is None or best["risk_reduction"] <= 1e-12:
                break

            # Teleport for baseline evaluation only; this baseline tests placement quality,
            # not navigation policy quality.
            self.agent_xy = best["xy"].copy()
            _, r, _, info = self.step(place_action_id)
            total_reward += r
            history.append({"chosen": best, "metrics": info["metrics"]})

        return total_reward, history


# # =============================================================================
# # 4. Export and plotting utilities
# # =============================================================================

# def save_csv(rows: List[Dict], path: Path):
#     if not rows:
#         return
#     keys = list(rows[0].keys())
#     with open(path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(rows)


# def plot_physical_arena(data: TrajectoryData, sensors: np.ndarray, arena_size: float, path: Path):
#     fig, ax = plt.subplots(figsize=(7, 7))
#     for p1, p2 in data.segments:
#         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.8, alpha=0.25)
#     if sensors.size > 0:
#         ax.scatter(sensors[:, 0], sensors[:, 1], marker="^", s=130, label="Sensors")
#     ax.set_xlim(0, arena_size)
#     ax.set_ylim(0, arena_size)
#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlabel("x position (km)")
#     ax.set_ylabel("y position (km)")
#     ax.set_title("Physical Trajectories and Final Sensors")
#     ax.legend()
#     plt.tight_layout()
#     fig.savefig(path, dpi=170)
#     plt.close(fig)


# def plot_intensity_3d(intensity: IntensityEstimate, path: Path):
#     X, Y = np.meshgrid(np.rad2deg(intensity.alpha_centers), intensity.p_centers, indexing="ij")
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(X, Y, intensity.lambda_plot, linewidth=0, antialiased=True, alpha=0.9)
#     ax.set_xlabel("alpha (deg)")
#     ax.set_ylabel("p (km)")
#     ax.set_zlabel("normalized lambda")
#     ax.set_title("MAP-smoothed Trajectory Intensity")
#     plt.tight_layout()
#     fig.savefig(path, dpi=170)
#     plt.close(fig)


# def plot_residual_3d(env: TrajectorySensorDeploymentEnv, path: Path):
#     R = env.current_metrics["residual_map"]
#     X, Y = np.meshgrid(np.rad2deg(env.alpha_centers), env.p_centers, indexing="ij")
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(X, Y, R, linewidth=0, antialiased=True, alpha=0.9)
#     ax.set_xlabel("alpha (deg)")
#     ax.set_ylabel("p (km)")
#     ax.set_zlabel("residual risk")
#     ax.set_title("Residual Missed-Trajectory Risk R(l,S)")
#     plt.tight_layout()
#     fig.savefig(path, dpi=170)
#     plt.close(fig)


# def plot_surface_3d(xs, ys, Z, title: str, zlabel: str, path: Path):
#     X, Y = np.meshgrid(xs, ys)
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.9)
#     ax.set_xlabel("x position (km)")
#     ax.set_ylabel("y position (km)")
#     ax.set_zlabel(zlabel)
#     ax.set_title(title)
#     plt.tight_layout()
#     fig.savefig(path, dpi=170)
#     plt.close(fig)


# # =============================================================================
# # 5. Demo
# # =============================================================================

# def demo(output_dir: str = "trajectory_rl_final_outputs"):
#     out = Path(output_dir)
#     out.mkdir(parents=True, exist_ok=True)

#     # Step 1: generate stochastic traffic trajectories
#     generator = TrajectoryGenerator(arena_size=10.0, seed=42)
#     data = generator.generate(n=350, traffic_type="multi_corridor")

#     # Step 2: estimate trajectory intensity in representation space
#     estimator = MAPLGCPEstimator(n_alpha=30, n_p=30, tau=2.0, kappa=1.2)
#     intensity = estimator.fit(data.rep_points)

#     # Step 3: build RL environment
#     env = TrajectorySensorDeploymentEnv(
#         intensity=intensity,
#         arena_size=10.0,
#         expected_traffic=10.0,
#         max_sensors=4,
#         max_steps=30,
#         motion_step=1.0,
#         rho=0.95,
#         sigma=0.45,
#         min_sensor_spacing=0.5,
#         seed=7,
#     )

#     # Step 4: validate mechanism, then expose starting state and action consequences
#     env.validate_mechanism()
#     env.reset(start_xy=[5.0, 5.0])
#     initial_obs = env.get_observation()
#     initial_action_rows = env.evaluate_action_consequences()
#     initial_risk_rows = env.trajectory_risk_table()
#     save_csv(initial_action_rows, out / "initial_action_consequence_table.csv")
#     save_csv(initial_risk_rows, out / "initial_trajectory_risk_table.csv")

#     # Step 5: explain reward surface from initial state
#     xs, ys, surfaces = env.counterfactual_placement_surface(nx=31, ny=31)

#     # Step 6: run two baseline episodes
#     random_rewards = []
#     random_final_metrics = []
#     for _ in range(20):
#         r, info = env.run_random_episode()
#         random_rewards.append(r)
#         random_final_metrics.append(info["metrics"])

#     greedy_reward, greedy_history = env.run_greedy_immediate_episode(start_xy=[5.0, 5.0])
#     final_sensors = np.asarray(env.sensors, dtype=float)
#     final_metrics = env.current_metrics

#     # Step 7: final tables and plots
#     save_csv(env.trajectory_risk_table(), out / "final_trajectory_risk_table.csv")
#     save_csv(env.evaluate_action_consequences(), out / "final_action_consequence_table.csv")

#     plot_intensity_3d(intensity, out / "01_3d_lgcp_intensity.png")
#     plot_surface_3d(xs, ys, surfaces["reward"], "Initial Counterfactual KPI Reward Surface", "reward", out / "02_3d_initial_reward_surface.png")
#     plot_surface_3d(xs, ys, surfaces["risk_reduction"], "Initial Counterfactual Risk-Reduction Surface", "risk reduction", out / "03_3d_initial_risk_reduction_surface.png")
#     plot_surface_3d(xs, ys, surfaces["worst_detection_gain"], "Initial Worst-Region Detection-Gain Surface", "worst detection gain", out / "04_3d_initial_worst_detection_gain_surface.png")
#     plot_surface_3d(xs, ys, surfaces["worst_risk_reduction"], "Initial Worst-Region Risk-Reduction Surface", "worst region risk reduction", out / "05_3d_initial_worst_risk_reduction_surface.png")
#     plot_residual_3d(env, out / "06_3d_final_residual_risk.png")
#     plot_physical_arena(data, final_sensors, env.arena_size, out / "07_physical_trajectories_and_greedy_sensors.png")

#     # Summary
#     avg_random_D = np.mean([m["detection_score"] for m in random_final_metrics])
#     avg_random_V = np.mean([m["void_probability"] for m in random_final_metrics])
#     avg_random_U = np.mean([m["expected_undetected"] for m in random_final_metrics])

#     summary = f"""
# Final trajectory-aware RL environment demo summary

# Trajectory generation:
#   Generated physical trajectories: {len(data.segments)}
#   Representation-space points: {len(data.rep_points)}

# MAP-smoothed Poisson intensity:
#   success: {intensity.success}
#   message: {intensity.message}
#   alpha bins: {len(intensity.alpha_centers)}
#   p bins: {len(intensity.p_centers)}

# State feature vector order:
#   0 agent_x_norm
#   1 agent_y_norm
#   2 sensors_used_norm
#   3 sensors_remaining_norm
#   4 step_norm
#   5 void_probability
#   6 normalized_log_void
#   7 detection_score
#   8 expected_undetected_norm
#   9 max_residual_risk_norm
#  10 residual_risk_entropy
#  11 worst_region_detection
#  12 worst_region_U_norm
#  13 mean_region_detection
#  14 dominant_alpha_norm
#  15 dominant_p_norm

# Initial continuous state:
#   {initial_obs["features"].tolist()}

# Reward definition:
#   reward =
#     normalized_risk_reduction
#   + fairness_weight * worst_region_detection_gain
#   - step_cost
#   - move_cost * movement_distance
#   - invalid_cost
#   - placement_cost
#   - redundancy_penalty
#   + terminal_reward if done

# Metric definitions:
#   U(S) = sum_l lambda_hat(l) * pi(l,S)
#   V(S) = exp(-U(S))
#   logV(S) = -U(S)
#   D(S) = 1 - U(S)/sum_l lambda_hat(l)
#   R(l,S) = lambda_hat(l) * pi(l,S)
#   U_k(S) = regional residual risk over coarse trajectory-region k
#   D_min(S) = worst-region detection score

# Random baseline over 20 episodes:
#   mean total reward: {float(np.mean(random_rewards)):.6f}
#   mean final detection_score: {float(avg_random_D):.6f}
#   mean final void_probability: {float(avg_random_V):.6f}
#   mean final expected_undetected: {float(avg_random_U):.6f}

# Greedy immediate-KPI baseline:
#   total reward: {float(greedy_reward):.6f}
#   final sensors:
# {final_sensors}
#   final detection_score: {final_metrics["detection_score"]:.6f}
#   final void_probability: {final_metrics["void_probability"]:.6f}
#   final expected_undetected: {final_metrics["expected_undetected"]:.6f}
#   final log_void: {final_metrics["log_void"]:.6f}
#   final worst_region_detection: {final_metrics["worst_region_detection"]:.6f}
#   final worst_region_U: {final_metrics["worst_region_U"]:.6f}

# Created key files:
#   initial_trajectory_risk_table.csv
#   initial_action_consequence_table.csv
#   final_trajectory_risk_table.csv
#   final_action_consequence_table.csv
#   01_3d_lgcp_intensity.png
#   02_3d_initial_reward_surface.png
#   03_3d_initial_risk_reduction_surface.png
#   04_3d_initial_worst_detection_gain_surface.png
#   05_3d_initial_worst_risk_reduction_surface.png
#   06_3d_final_residual_risk.png
#   07_physical_trajectories_and_greedy_sensors.png
# """
#     (out / "summary.txt").write_text(summary.strip())

#     print(summary)
#     print(f"\nAll outputs saved to: {out}")


# if __name__ == "__main__":
#     demo()