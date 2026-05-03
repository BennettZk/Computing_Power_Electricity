from __future__ import annotations

import numpy as np
import pandas as pd

from models.resource import ResourcePool
from models.task import Task
from optimizers.common import SingleObjectiveRunResult, build_bounds, evaluate_weighted_fitness


def _evaluate_swarm(
    positions: np.ndarray,
    bounds,
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
) -> tuple[np.ndarray, list[tuple]]:
    fitness_values: list[float] = []
    evaluated: list[tuple] = []
    for position in positions:
        fitness, schedule, metrics = evaluate_weighted_fitness(
            vector=position,
            bounds=bounds,
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
        )
        fitness_values.append(fitness)
        evaluated.append((fitness, schedule, metrics))
    return np.asarray(fitness_values, dtype=float), evaluated


def run_pso(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
) -> SingleObjectiveRunResult:
    """离散粒子群优化：连续位置更新，评价时按共享编码 round/clip。"""
    optimizer_cfg = experiment_cfg["optimizer"]
    bounds = build_bounds(resource_pool, base_cfg, experiment_cfg)
    n_var = bounds.xl.size
    swarm_size = max(4, int(optimizer_cfg.get("pso_swarm_size", 30)))
    n_iter = max(1, int(optimizer_cfg.get("pso_n_iter", 20)))
    inertia = float(optimizer_cfg.get("pso_w", 0.7))
    cognitive = float(optimizer_cfg.get("pso_c1", 1.5))
    social = float(optimizer_cfg.get("pso_c2", 1.5))
    rng = np.random.default_rng(int(experiment_cfg["seed"]) + 202)

    xl = bounds.xl.astype(float)
    xu = bounds.xu.astype(float)
    span = np.maximum(xu - xl, 1.0)
    positions = rng.uniform(xl, xu + 1.0, size=(swarm_size, n_var))
    positions = np.clip(positions, xl, xu)
    velocities = rng.uniform(-0.25 * span, 0.25 * span, size=(swarm_size, n_var))

    pbest_positions = positions.copy()
    pbest_fitness = np.full(swarm_size, float("inf"), dtype=float)
    global_best_position = positions[0].copy()
    global_best_fitness = float("inf")
    global_best_schedule = None
    global_best_metrics = None
    convergence_rows: list[dict] = []

    for iteration in range(n_iter):
        fitness_values, evaluated = _evaluate_swarm(
            positions=positions,
            bounds=bounds,
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
        )

        improved = fitness_values < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness_values[improved]

        iteration_best_idx = int(np.argmin(fitness_values))
        if fitness_values[iteration_best_idx] < global_best_fitness:
            global_best_fitness = float(fitness_values[iteration_best_idx])
            global_best_position = positions[iteration_best_idx].copy()
            global_best_schedule = evaluated[iteration_best_idx][1]
            global_best_metrics = evaluated[iteration_best_idx][2]

        convergence_rows.append(
            {
                "generation": iteration,
                "best_fitness": global_best_fitness,
                "iteration_best_fitness": float(fitness_values[iteration_best_idx]),
                "avg_fitness": float(np.mean(fitness_values)),
            }
        )

        r1 = rng.random(size=(swarm_size, n_var))
        r2 = rng.random(size=(swarm_size, n_var))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest_positions - positions)
            + social * r2 * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -span, span)
        positions = np.clip(positions + velocities, xl, xu)

    if global_best_schedule is None or global_best_metrics is None:
        raise RuntimeError("PSO failed to evaluate any particle.")

    metrics_dict = {**global_best_metrics.to_dict(), "fitness": global_best_fitness}
    return SingleObjectiveRunResult(
        best_schedule=global_best_schedule,
        best_metrics=metrics_dict,
        convergence_df=pd.DataFrame(convergence_rows),
    )
