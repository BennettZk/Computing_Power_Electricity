from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceType:
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
        return max(0, min(self.count, int(active_servers)))

    def compute_capacity(self, active_servers: int) -> float:
        return self.clamp_active(active_servers) * self.queue_service_rate

    def memory_capacity_total(self, active_servers: int) -> float:
        return self.clamp_active(active_servers) * self.memory_capacity

    def bandwidth_capacity_total(self, active_servers: int) -> float:
        return self.clamp_active(active_servers) * self.bandwidth_capacity


@dataclass(frozen=True)
class ResourcePool:
    cpu: ResourceType
    gpu: ResourceType

    def as_dict(self) -> dict[str, ResourceType]:
        return {"cpu": self.cpu, "gpu": self.gpu}

    def server_limits(self) -> dict[str, int]:
        return {name: resource.count for name, resource in self.as_dict().items()}
