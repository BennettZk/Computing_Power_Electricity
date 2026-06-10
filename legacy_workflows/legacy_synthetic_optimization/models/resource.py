from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceType:
    """单类异构资源的静态参数，例如 CPU 型服务器或 GPU 型服务器。"""

    server_type: str
    count: int
    max_compute: float
    idle_power: float
    peak_power: float
    memory_capacity: float
    bandwidth_capacity: float
    queue_service_rate: float
    cooling_coeff: float

    def clamp_active(self, active_servers: int) -> int:
        """将开机台数限制在该资源池可用服务器数量范围内。"""
        return max(0, min(self.count, int(active_servers)))

    def compute_capacity(self, active_servers: int) -> float:
        """估算当前开机台数对应的小时级服务能力。"""
        return self.clamp_active(active_servers) * self.queue_service_rate

    def memory_capacity_total(self, active_servers: int) -> float:
        """估算当前开机服务器可提供的总内存容量。"""
        return self.clamp_active(active_servers) * self.memory_capacity

    def bandwidth_capacity_total(self, active_servers: int) -> float:
        """估算当前开机服务器可提供的总带宽容量。"""
        return self.clamp_active(active_servers) * self.bandwidth_capacity


@dataclass(frozen=True)
class ResourcePool:
    """单数据中心内的异构资源池，目前包含 CPU 与 GPU 两类资源。"""

    cpu: ResourceType
    gpu: ResourceType

    def as_dict(self) -> dict[str, ResourceType]:
        """便于按资源类型名称索引资源参数。"""
        return {"cpu": self.cpu, "gpu": self.gpu}

    def server_limits(self) -> dict[str, int]:
        """返回 CPU 与 GPU 资源的服务器数量上限。"""
        return {name: resource.count for name, resource in self.as_dict().items()}
