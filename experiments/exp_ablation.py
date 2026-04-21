from __future__ import annotations

import numpy as np
import pandas as pd

from models.objective import SchedulePlan, simulate_schedule
from models.resource import ResourcePool
from models.task import Task
from schedulers.homogeneous_baseline import build_homogeneous_schedule
from utils.metrics import summarize_result_rows


def run_ablation_experiment(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
    proposed_schedule: SchedulePlan,
    homogeneous_schedule: SchedulePlan | None = None,
) -> pd.DataFrame:
    """真实消融实验：在同一批任务上关闭 Proposed 的关键机制并重新评价。"""
    hours = int(base_cfg["time"]["hours"])
    no_defer_schedule = SchedulePlan(
        cpu_servers=proposed_schedule.cpu_servers.copy(),
        gpu_servers=proposed_schedule.gpu_servers.copy(),
        defer_ratio=np.zeros(hours, dtype=float),
        migration_ratio=proposed_schedule.migration_ratio.copy(),
    )
    no_spatial_schedule = SchedulePlan(
        cpu_servers=proposed_schedule.cpu_servers.copy(),
        gpu_servers=proposed_schedule.gpu_servers.copy(),
        defer_ratio=proposed_schedule.defer_ratio.copy(),
        migration_ratio=np.zeros(hours, dtype=float),
    )
    if homogeneous_schedule is None:
        homogeneous_schedule = build_homogeneous_schedule(hourly_df, resource_pool, base_cfg, experiment_cfg)

    ablation_runs = [
        ("完整Proposed", proposed_schedule, "proposed"),
        ("无空间迁移", no_spatial_schedule, "proposed"),
        ("无时间迁移", no_defer_schedule, "proposed"),
        ("无异构感知", homogeneous_schedule, "fcfs"),
        ("无优先级调度", proposed_schedule, "no_priority"),
    ]

    rows: list[dict] = []
    for scenario, schedule, mode in ablation_runs:
        metrics = simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode=mode,
        )
        rows.append({"algorithm": scenario, **metrics.to_dict()})

    return summarize_result_rows(rows)


def run_ablation() -> pd.DataFrame:
    """独立运行入口，便于单独调试消融实验。"""
    from experiments.exp_main import run_main_experiment

    _, artifacts = run_main_experiment()
    configs = artifacts["configs"]
    return run_ablation_experiment(
        hourly_df=artifacts["hourly_df"],
        tasks=artifacts["tasks"],
        resource_pool=artifacts["resource_pool"],
        base_cfg=configs["base"],
        price_cfg=configs["price"],
        experiment_cfg=configs["experiment"],
        proposed_schedule=artifacts["proposed_result"].best_schedule,
    )
