from __future__ import annotations

import numpy as np
import pandas as pd

from models.objective import SchedulePlan
from models.resource import ResourcePool
from models.task import Task, aggregate_tasks_by_slot


def build_price_only_schedule(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
) -> SchedulePlan:
    hours = int(base_cfg["time"]["hours"])
    aggregated = aggregate_tasks_by_slot(tasks, hours)
    high_threshold = float(hourly_df["price"].quantile(price_cfg["high_price_quantile"]))
    low_threshold = float(hourly_df["price"].quantile(price_cfg["low_price_quantile"]))
    discount = float(experiment_cfg["dispatch"]["price_response_discount"])
    release_bonus = float(experiment_cfg["dispatch"]["price_response_release_bonus"])

    cpu_base = aggregated["cpu_task_count"] / resource_pool.cpu.queue_service_rate
    gpu_base = aggregated["gpu_task_count"] / resource_pool.gpu.queue_service_rate

    cpu_servers = []
    gpu_servers = []
    defer_ratio = []

    for row, cpu_need, gpu_need in zip(hourly_df.itertuples(index=False), cpu_base, gpu_base):
        price = float(row.price)
        sensitive = float(aggregated.loc[int(row.hour), "delay_sensitive_count"])
        tolerant = float(aggregated.loc[int(row.hour), "delay_tolerant_count"] + aggregated.loc[int(row.hour), "token_batch_count"])
        urgent_cpu = sensitive / max(resource_pool.cpu.queue_service_rate, 1.0)
        urgent_gpu = (aggregated.loc[int(row.hour), "token_batch_count"] * 0.8) / max(resource_pool.gpu.queue_service_rate, 1.0)

        if price >= high_threshold:
            cpu_value = np.ceil(urgent_cpu + max(0.0, cpu_need - urgent_cpu) * (1.0 - discount))
            gpu_value = np.ceil(urgent_gpu + max(0.0, gpu_need - urgent_gpu) * (1.0 - discount))
            defer = 0.6 if tolerant > 0 else 0.0
        elif price <= low_threshold:
            cpu_value = np.ceil(cpu_need * (1.0 + release_bonus))
            gpu_value = np.ceil(gpu_need * (1.0 + release_bonus))
            defer = 0.0
        else:
            cpu_value = np.ceil(cpu_need)
            gpu_value = np.ceil(gpu_need)
            defer = 0.2 if tolerant > 0 else 0.0

        cpu_servers.append(int(np.clip(cpu_value, experiment_cfg["optimizer"]["cpu_min_active"], resource_pool.cpu.count)))
        gpu_servers.append(int(np.clip(gpu_value, experiment_cfg["optimizer"]["gpu_min_active"], resource_pool.gpu.count)))
        defer_ratio.append(float(defer))

    return SchedulePlan(
        cpu_servers=np.array(cpu_servers, dtype=int),
        gpu_servers=np.array(gpu_servers, dtype=int),
        defer_ratio=np.array(defer_ratio, dtype=float),
    )
