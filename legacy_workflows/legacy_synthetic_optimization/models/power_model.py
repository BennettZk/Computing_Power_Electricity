from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from models.resource import ResourcePool, ResourceType


def estimate_utilization(arrival_rate: float, service_rate_per_server: float, servers: int) -> float:
    """用负载/服务能力估算资源利用率，并截断到 [0, 1]。"""
    if servers <= 0 or service_rate_per_server <= 0:
        return 0.0
    capacity = servers * service_rate_per_server
    return float(np.clip(arrival_rate / capacity, 0.0, 1.0))


def hourly_it_power_kw(
    arrival_rate: float,
    service_rate_per_server: float,
    servers: int,
    idle_power_kw: float,
    peak_power_kw: float,
) -> float:
    """保留原同构服务器功率接口：单机功率按空闲功率到峰值功率线性插值。"""
    utilization = estimate_utilization(arrival_rate, service_rate_per_server, servers)
    per_server_power = idle_power_kw + utilization * (peak_power_kw - idle_power_kw)
    return max(0.0, servers * per_server_power)


def hourly_total_power_kw(
    arrival_rate: float,
    service_rate_per_server: float,
    servers: int,
    idle_power_kw: float,
    peak_power_kw: float,
    pue: float,
) -> float:
    """计算单个小时的异构资源总功率。"""
    return hourly_it_power_kw(
        arrival_rate=arrival_rate,
        service_rate_per_server=service_rate_per_server,
        servers=servers,
        idle_power_kw=idle_power_kw,
        peak_power_kw=peak_power_kw,
    ) * pue


def hourly_resource_it_power_kw(load_demand: float, active_servers: int, resource: ResourceType) -> float:
    """异构版本的 IT 功率计算，按 CPU/GPU 资源类型分别估算。"""
    utilization = estimate_utilization(load_demand, resource.queue_service_rate, active_servers)
    per_server_power = resource.idle_power + utilization * (resource.peak_power - resource.idle_power)
    return resource.clamp_active(active_servers) * per_server_power


def hourly_heterogeneous_power_kw(
    resource_pool: ResourcePool,
    active_servers: Mapping[str, int],
    served_load: Mapping[str, float],
    fixed_power_kw: float,
    cooling_base_coeff: float,
) -> dict[str, float]:
    """计算单个时隙的总功率：CPU IT 功率 + GPU IT 功率 + 制冷功率 + 固定功率。"""
    cpu_power = hourly_resource_it_power_kw(served_load.get("cpu", 0.0), active_servers.get("cpu", 0), resource_pool.cpu)
    gpu_power = hourly_resource_it_power_kw(served_load.get("gpu", 0.0), active_servers.get("gpu", 0), resource_pool.gpu)
    cooling_power = (
        cpu_power * resource_pool.cpu.cooling_coeff
        + gpu_power * resource_pool.gpu.cooling_coeff
        + cooling_base_coeff
    )
    total = cpu_power + gpu_power + cooling_power + fixed_power_kw
    return {
        "cpu_it_power_kw": cpu_power,
        "gpu_it_power_kw": gpu_power,
        "cooling_power_kw": cooling_power,
        "fixed_power_kw": fixed_power_kw,
        "total_power_kw": total,
    }


def _sum_weighted(values: Sequence[float], weights: Sequence[float], delta_t_hours: float) -> float:
    """按小时权重累加功率序列对应的成本或排放量。"""
    return float(sum(float(v) * float(w) * delta_t_hours for v, w in zip(values, weights)))


def total_energy_cost(
    arrival_rates=None,
    prices=None,
    server_plan=None,
    service_rate_per_server: float | None = None,
    idle_power_kw: float | None = None,
    peak_power_kw: float | None = None,
    pue: float | None = None,
    delta_t_hours: float = 1.0,
    hourly_power_kw: Sequence[float] | None = None,
) -> float:
    """计算总电费；既支持新异构功率序列，也兼容原同构服务器参数。"""
    if hourly_power_kw is not None and prices is not None:
        return _sum_weighted(hourly_power_kw, prices, delta_t_hours)

    if any(value is None for value in [arrival_rates, prices, server_plan, service_rate_per_server, idle_power_kw, peak_power_kw, pue]):
        raise ValueError("Either pass hourly_power_kw with prices or the homogeneous power arguments.")

    total_cost = 0.0
    for lam, price, servers in zip(arrival_rates, prices, server_plan):
        power_kw = hourly_total_power_kw(
            arrival_rate=float(lam),
            service_rate_per_server=float(service_rate_per_server),
            servers=int(servers),
            idle_power_kw=float(idle_power_kw),
            peak_power_kw=float(peak_power_kw),
            pue=float(pue),
        )
        total_cost += power_kw * delta_t_hours * float(price)
    return float(total_cost)


def total_carbon_emission(
    arrival_rates=None,
    carbon_factors=None,
    server_plan=None,
    service_rate_per_server: float | None = None,
    idle_power_kw: float | None = None,
    peak_power_kw: float | None = None,
    pue: float | None = None,
    delta_t_hours: float = 1.0,
    hourly_power_kw: Sequence[float] | None = None,
) -> float:
    """计算碳排放量；异构模式下直接使用每小时总功率与碳因子相乘。"""
    if hourly_power_kw is not None and carbon_factors is not None:
        return _sum_weighted(hourly_power_kw, carbon_factors, delta_t_hours)

    if any(value is None for value in [arrival_rates, carbon_factors, server_plan, service_rate_per_server, idle_power_kw, peak_power_kw, pue]):
        raise ValueError("Either pass hourly_power_kw with carbon_factors or the homogeneous power arguments.")

    total_carbon = 0.0
    for lam, cf, servers in zip(arrival_rates, carbon_factors, server_plan):
        power_kw = hourly_total_power_kw(
            arrival_rate=float(lam),
            service_rate_per_server=float(service_rate_per_server),
            servers=int(servers),
            idle_power_kw=float(idle_power_kw),
            peak_power_kw=float(peak_power_kw),
            pue=float(pue),
        )
        total_carbon += power_kw * delta_t_hours * float(cf)
    return float(total_carbon)
