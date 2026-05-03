from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.objective import SchedulePlan, SimulationResult, simulate_schedule
from models.resource import ResourcePool
from models.task import Task
from schedulers.proposed_scheduler import run_proposed_scheduler


@dataclass
class RollingExperimentResult:
    """滚动优化实验输出：拼接后的调度方案、完整指标和结果表。"""

    rolling_schedule: SchedulePlan
    rolling_metrics: SimulationResult
    static_metrics: SimulationResult
    results_df: pd.DataFrame


def _window_base_cfg(base_cfg: dict, start_hour: int, window_hours: int) -> dict:
    cfg = {
        **base_cfg,
        "time": {**base_cfg["time"], "hours": window_hours},
        "constraints": {**base_cfg["constraints"]},
        "power": {**base_cfg["power"]},
        "task_defaults": {**base_cfg["task_defaults"]},
        "migration": {**base_cfg.get("migration", {})},
    }
    cfg["constraints"]["peak_hours"] = [
        int(hour - start_hour)
        for hour in base_cfg["constraints"].get("peak_hours", [])
        if start_hour <= int(hour) < start_hour + window_hours
    ]
    return cfg


def _window_experiment_cfg(experiment_cfg: dict, start_hour: int) -> dict:
    optimizer_cfg = experiment_cfg["optimizer"]
    rolling_optimizer = {
        **optimizer_cfg,
        "pop_size": int(optimizer_cfg.get("rolling_pop_size", 20)),
        "n_gen": int(optimizer_cfg.get("rolling_n_gen", 8)),
    }
    return {
        **experiment_cfg,
        "seed": int(experiment_cfg["seed"]) + start_hour,
        "optimizer": rolling_optimizer,
    }


def _slice_hourly_df(hourly_df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    window_df = hourly_df.iloc[start_hour:end_hour].copy().reset_index(drop=True)
    window_df["hour"] = np.arange(len(window_df), dtype=int)
    return window_df


def _slice_window_tasks(tasks: list[Task], start_hour: int, end_hour: int) -> list[Task]:
    """提取当前队列和未来窗口任务，并将到达/截止时隙改写为窗口内相对时间。"""
    window_tasks: list[Task] = []
    for task in tasks:
        if int(task.arrival_time) >= end_hour or int(task.deadline_slot) < start_hour:
            continue
        arrival = max(int(task.arrival_time), start_hour) - start_hour
        deadline = min(int(task.deadline_slot), end_hour - 1) - start_hour
        deadline = max(arrival, deadline)
        window_tasks.append(
            Task(
                task_id=task.task_id,
                arrival_time=arrival,
                task_type=task.task_type,
                cpu_demand=task.cpu_demand,
                gpu_demand=task.gpu_demand,
                memory_demand=task.memory_demand,
                bandwidth_demand=task.bandwidth_demand,
                token_amount=task.token_amount,
                deadline=deadline,
                priority=task.priority,
                migratable=task.migratable,
                migration_cost_weight=task.migration_cost_weight,
                migration_delay_penalty=task.migration_delay_penalty,
            )
        )
    return window_tasks


def _metrics_row(
    algorithm: str,
    scope: str,
    metrics: SimulationResult,
    window_start: int | None = None,
    window_end: int | None = None,
    optimized_task_count: int | None = None,
) -> dict:
    return {
        "algorithm": algorithm,
        "scope": scope,
        "window_start": window_start,
        "window_end": window_end,
        "optimized_task_count": optimized_task_count,
        "total_energy_kwh": metrics.total_energy_kwh,
        "total_cost": metrics.total_cost,
        "avg_delay_hours": metrics.avg_delay_hours,
        "sla_violation_rate": metrics.sla_violation_rate,
        "completion_rate": metrics.completion_rate,
        "remote_task_count": metrics.remote_task_count,
        "remote_completion_rate": metrics.remote_completion_rate,
        "remote_cost": metrics.remote_cost,
        "remote_energy_kwh": metrics.remote_energy_kwh,
        "peak_valley_gap_kw": metrics.peak_valley_gap_kw,
    }


def run_rolling_experiment(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
    static_schedule: SchedulePlan,
    window_size: int = 4,
) -> RollingExperimentResult:
    """Rolling-Proposed：每个 4 小时窗口重新优化，拼接后与 Static-Proposed 对比。"""
    hours = int(base_cfg["time"]["hours"])
    cpu_plan = np.zeros(hours, dtype=int)
    gpu_plan = np.zeros(hours, dtype=int)
    defer_plan = np.zeros(hours, dtype=float)
    migration_plan = np.zeros(hours, dtype=float)
    rows: list[dict] = []

    for start_hour in range(0, hours, window_size):
        end_hour = min(start_hour + window_size, hours)
        window_hours = end_hour - start_hour
        window_df = _slice_hourly_df(hourly_df, start_hour, end_hour)
        window_tasks = _slice_window_tasks(tasks, start_hour, end_hour)
        window_base_cfg = _window_base_cfg(base_cfg, start_hour, window_hours)
        window_experiment_cfg = _window_experiment_cfg(experiment_cfg, start_hour)

        window_result = run_proposed_scheduler(
            hourly_df=window_df,
            tasks=window_tasks,
            resource_pool=resource_pool,
            base_cfg=window_base_cfg,
            price_cfg=price_cfg,
            experiment_cfg=window_experiment_cfg,
        )
        cpu_plan[start_hour:end_hour] = window_result.best_schedule.cpu_servers[:window_hours]
        gpu_plan[start_hour:end_hour] = window_result.best_schedule.gpu_servers[:window_hours]
        defer_plan[start_hour:end_hour] = window_result.best_schedule.defer_ratio[:window_hours]
        migration_plan[start_hour:end_hour] = window_result.best_schedule.migration_ratio[:window_hours]

        window_metrics = simulate_schedule(
            hourly_df=window_df,
            tasks=window_tasks,
            resource_pool=resource_pool,
            schedule=window_result.best_schedule,
            base_cfg=window_base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="proposed",
        )
        rows.append(
            _metrics_row(
                algorithm="Rolling-Proposed",
                scope="window",
                metrics=window_metrics,
                window_start=start_hour,
                window_end=end_hour - 1,
                optimized_task_count=len(window_tasks),
            )
        )

    rolling_schedule = SchedulePlan(
        cpu_servers=cpu_plan,
        gpu_servers=gpu_plan,
        defer_ratio=defer_plan,
        migration_ratio=migration_plan,
    )
    rolling_metrics = simulate_schedule(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        schedule=rolling_schedule,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        dispatch_mode="proposed",
    )
    static_metrics = simulate_schedule(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        schedule=static_schedule,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        dispatch_mode="proposed",
    )

    rows.insert(0, _metrics_row("Static-Proposed", "overall", static_metrics))
    rows.insert(1, _metrics_row("Rolling-Proposed", "overall", rolling_metrics))
    return RollingExperimentResult(
        rolling_schedule=rolling_schedule,
        rolling_metrics=rolling_metrics,
        static_metrics=static_metrics,
        results_df=pd.DataFrame(rows),
    )


def run_rolling() -> RollingExperimentResult:
    """独立运行入口，便于单独调试滚动窗口实验。"""
    from utils.io_utils import ensure_inputs, load_all_configs, load_resource_pool

    configs = load_all_configs()
    hourly_df, tasks = ensure_inputs(configs)
    resource_pool = load_resource_pool(configs["resource"])
    proposed_result = run_proposed_scheduler(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=configs["base"],
        price_cfg=configs["price"],
        experiment_cfg=configs["experiment"],
    )
    return run_rolling_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=configs["base"],
        price_cfg=configs["price"],
        experiment_cfg=configs["experiment"],
        static_schedule=proposed_result.best_schedule,
    )
