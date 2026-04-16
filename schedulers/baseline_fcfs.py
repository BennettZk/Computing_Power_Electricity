from __future__ import annotations

import numpy as np

from models.objective import SchedulePlan
from models.resource import ResourcePool
from models.task import Task, aggregate_tasks_by_slot


def build_fcfs_schedule(tasks: list[Task], resource_pool: ResourcePool, base_cfg: dict, experiment_cfg: dict) -> SchedulePlan:
    hours = int(base_cfg["time"]["hours"])
    aggregated = aggregate_tasks_by_slot(tasks, hours)
    headroom = float(experiment_cfg["dispatch"]["fcfs_headroom"])

    cpu_servers = np.ceil(aggregated["cpu_task_count"] / resource_pool.cpu.queue_service_rate * headroom).astype(int)
    gpu_servers = np.ceil(aggregated["gpu_task_count"] / resource_pool.gpu.queue_service_rate * headroom).astype(int)

    cpu_servers = np.clip(cpu_servers, int(experiment_cfg["optimizer"]["cpu_min_active"]), resource_pool.cpu.count)
    gpu_servers = np.clip(gpu_servers, int(experiment_cfg["optimizer"]["gpu_min_active"]), resource_pool.gpu.count)

    return SchedulePlan(
        cpu_servers=cpu_servers.to_numpy(dtype=int),
        gpu_servers=gpu_servers.to_numpy(dtype=int),
        defer_ratio=np.zeros(hours, dtype=float),
    )
