from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.objective import SchedulePlan, SimulationResult, simulate_schedule
from models.resource import ResourcePool
from models.task import Task


@dataclass(frozen=True)
class DecisionBounds:
    """共享的调度变量边界：CPU、GPU、时间迁移比例等级、空间迁移比例等级。"""

    xl: np.ndarray
    xu: np.ndarray
    hours: int
    defer_levels: int
    migration_levels: int


@dataclass
class SingleObjectiveRunResult:
    """GA/PSO 单目标优化结果，字段尽量与 NSGA-II 结果保持一致。"""

    best_schedule: SchedulePlan
    best_metrics: dict
    convergence_df: pd.DataFrame


def build_bounds(resource_pool: ResourcePool, base_cfg: dict, experiment_cfg: dict) -> DecisionBounds:
    optimizer_cfg = experiment_cfg["optimizer"]
    hours = int(base_cfg["time"]["hours"])
    defer_levels = max(1, int(optimizer_cfg["defer_ratio_levels"]))
    migration_levels = max(1, int(optimizer_cfg.get("migration_ratio_levels", defer_levels)))
    migration_cfg = base_cfg.get("migration", {})
    max_migration_ratio = float(migration_cfg.get("max_migration_ratio", 0.0))
    max_migration_level = int(round(max_migration_ratio * migration_levels))
    max_migration_level = max(0, min(max_migration_level, migration_levels))

    xl = np.concatenate(
        [
            np.full(hours, int(optimizer_cfg["cpu_min_active"])),
            np.full(hours, int(optimizer_cfg["gpu_min_active"])),
            np.zeros(hours, dtype=int),
            np.zeros(hours, dtype=int),
        ]
    )
    xu = np.concatenate(
        [
            np.full(hours, int(resource_pool.cpu.count)),
            np.full(hours, int(resource_pool.gpu.count)),
            np.full(hours, defer_levels),
            np.full(hours, max_migration_level),
        ]
    )
    return DecisionBounds(xl=xl.astype(int), xu=xu.astype(int), hours=hours, defer_levels=defer_levels, migration_levels=migration_levels)


def decode_schedule(vector: np.ndarray, bounds: DecisionBounds) -> SchedulePlan:
    values = np.rint(np.asarray(vector, dtype=float)).astype(int)
    values = np.clip(values, bounds.xl, bounds.xu)
    hours = bounds.hours
    cpu = values[:hours].astype(int)
    gpu = values[hours : 2 * hours].astype(int)
    defer = values[2 * hours : 3 * hours].astype(float) / bounds.defer_levels
    migration = values[3 * hours : 4 * hours].astype(float) / bounds.migration_levels
    return SchedulePlan(cpu_servers=cpu, gpu_servers=gpu, defer_ratio=defer, migration_ratio=migration)


def weighted_fitness_from_metrics(metrics: SimulationResult) -> float:
    return float(
        metrics.total_cost
        + 1000.0 * metrics.avg_delay_hours
        + 5000.0 * metrics.sla_violation_rate
        + 200.0 * metrics.load_imbalance
        + metrics.objective_penalty
    )


def evaluate_weighted_fitness(
    vector: np.ndarray,
    bounds: DecisionBounds,
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
) -> tuple[float, SchedulePlan, SimulationResult]:
    schedule = decode_schedule(vector, bounds)
    metrics = simulate_schedule(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        schedule=schedule,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        dispatch_mode="proposed",
    )
    return weighted_fitness_from_metrics(metrics), schedule, metrics
