from dataclasses import dataclass

from utils.io_utils import load_all_configs


@dataclass(frozen=True)
class SystemConfig:
    hours: int
    delta_t_hours: float
    min_servers: int
    max_servers: int
    service_rate_per_server: float
    server_idle_power_kw: float
    server_peak_power_kw: float
    pue: float
    avg_tokens_per_request: float
    max_avg_delay_hours: float
    infeasible_penalty: float

    @classmethod
    def from_project_config(cls) -> "SystemConfig":
        configs = load_all_configs()
        base_cfg = configs["base"]
        cpu_resource = next(item for item in configs["resource"]["resource_types"] if item["server_type"] == "cpu")
        return cls(
            hours=int(base_cfg["time"]["hours"]),
            delta_t_hours=float(base_cfg["time"]["slot_hours"]),
            min_servers=1,
            max_servers=int(cpu_resource["count"]),
            service_rate_per_server=float(cpu_resource["queue_service_rate"]),
            server_idle_power_kw=float(cpu_resource["idle_power"]),
            server_peak_power_kw=float(cpu_resource["peak_power"]),
            pue=1.25,
            avg_tokens_per_request=800.0,
            max_avg_delay_hours=float(base_cfg["constraints"]["max_avg_delay_hours"]),
            infeasible_penalty=float(base_cfg["constraints"]["infeasible_penalty"]),
        )


def load_system_config() -> SystemConfig:
    return SystemConfig.from_project_config()
