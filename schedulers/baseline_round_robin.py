from __future__ import annotations

import numpy as np

from models.objective import SchedulePlan
from models.resource import ResourcePool


def build_round_robin_schedule(hours: int, resource_pool: ResourcePool) -> SchedulePlan:
    cpu_servers = np.array([(hour % max(resource_pool.cpu.count, 1)) + 1 for hour in range(hours)], dtype=int)
    gpu_servers = np.array([hour % max(resource_pool.gpu.count, 1) for hour in range(hours)], dtype=int)
    defer_ratio = np.zeros(hours, dtype=float)
    migration_ratio = np.zeros(hours, dtype=float)
    return SchedulePlan(cpu_servers=cpu_servers, gpu_servers=gpu_servers, defer_ratio=defer_ratio, migration_ratio=migration_ratio)
