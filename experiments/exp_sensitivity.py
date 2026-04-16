from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pandas as pd

from models.resource import ResourcePool
from models.task import Task, generate_synthetic_tasks, load_tasks_csv
from schedulers.proposed_scheduler import run_proposed_scheduler


def _lightweight_experiment_cfg(experiment_cfg: dict) -> dict:
    """灵敏度实验使用较小种群和代数，保证 main.py 不会运行过久。"""
    cfg = {
        **experiment_cfg,
        "optimizer": {**experiment_cfg["optimizer"], "pop_size": 24, "n_gen": 8},
    }
    return cfg


def _generate_scaled_tasks(hourly_df: pd.DataFrame, base_cfg: dict, seed: int, scale: float, name: str) -> list[Task]:
    """按负载倍率重新生成任务序列，而不是直接篡改结果数值。"""
    scaled_df = hourly_df.copy()
    scaled_df["arrival_rate"] = (scaled_df["arrival_rate"] * scale).round().clip(lower=1).astype(int)
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp_dir:
        output_path = Path(tmp_dir) / "tasks.csv"
        generate_synthetic_tasks(
            hourly_df=scaled_df,
            output_path=output_path,
            seed=seed,
            max_delay_slots_tolerant=int(base_cfg["constraints"]["max_delay_slots_tolerant"]),
            max_delay_slots_token=int(base_cfg["constraints"]["max_delay_slots_token"]),
        )
        return load_tasks_csv(output_path)


def _halve_gpu_pool(resource_pool: ResourcePool) -> ResourcePool:
    """构造 GPU 紧缺场景：GPU 数量减半，但 CPU 配置不变。"""
    gpu_count = max(1, resource_pool.gpu.count // 2)
    return ResourcePool(cpu=resource_pool.cpu, gpu=replace(resource_pool.gpu, count=gpu_count))


def _tighten_power_limit(base_cfg: dict, ratio: float) -> dict:
    """构造功率上限收紧场景。"""
    cfg = {
        **base_cfg,
        "constraints": {**base_cfg["constraints"]},
    }
    cfg["constraints"]["power_limit_kw"] = float(base_cfg["constraints"]["power_limit_kw"]) * ratio
    return cfg


def run_sensitivity_experiment(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
) -> pd.DataFrame:
    """真实灵敏度实验：改变负载、GPU 数量和功率约束后重新运行 Proposed。"""
    light_cfg = _lightweight_experiment_cfg(experiment_cfg)
    seed = int(experiment_cfg["seed"])

    scenarios = [
        ("基准场景", hourly_df, tasks, resource_pool, base_cfg),
        ("低负载0.8x", hourly_df, _generate_scaled_tasks(hourly_df, base_cfg, seed + 1, 0.8, "load_08"), resource_pool, base_cfg),
        ("高负载1.2x", hourly_df, _generate_scaled_tasks(hourly_df, base_cfg, seed + 2, 1.2, "load_12"), resource_pool, base_cfg),
        ("GPU数量减半", hourly_df, tasks, _halve_gpu_pool(resource_pool), base_cfg),
        ("功率上限收紧", hourly_df, tasks, resource_pool, _tighten_power_limit(base_cfg, 0.85)),
    ]

    rows: list[dict] = []
    for scenario, scenario_hourly_df, scenario_tasks, scenario_pool, scenario_base_cfg in scenarios:
        result = run_proposed_scheduler(
            hourly_df=scenario_hourly_df,
            tasks=scenario_tasks,
            resource_pool=scenario_pool,
            base_cfg=scenario_base_cfg,
            price_cfg=price_cfg,
            experiment_cfg=light_cfg,
        )
        metrics = result.best_metrics
        rows.append(
            {
                "scenario": scenario,
                "total_energy_kwh": metrics["total_energy_kwh"],
                "total_cost": metrics["total_cost"],
                "avg_delay_hours": metrics["avg_delay_hours"],
                "sla_violation_rate": metrics["sla_violation_rate"],
                "completion_rate": metrics["completion_rate"],
                "avg_cpu_utilization": metrics["avg_cpu_utilization"],
                "avg_gpu_utilization": metrics["avg_gpu_utilization"],
                "load_imbalance": metrics["load_imbalance"],
                "peak_valley_gap_kw": metrics["peak_valley_gap_kw"],
                "unit_token_energy_kwh_per_million": metrics["unit_token_energy_kwh_per_million"],
                "unit_token_cost_per_million": metrics["unit_token_cost_per_million"],
            }
        )

    return pd.DataFrame(rows)


def run_sensitivity() -> pd.DataFrame:
    """独立运行入口，便于单独调试灵敏度实验。"""
    from utils.io_utils import ensure_inputs, load_all_configs, load_resource_pool

    configs = load_all_configs()
    hourly_df, tasks = ensure_inputs(configs)
    return run_sensitivity_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=load_resource_pool(configs["resource"]),
        base_cfg=configs["base"],
        price_cfg=configs["price"],
        experiment_cfg=configs["experiment"],
    )
