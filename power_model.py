import numpy as np


def estimate_utilization(
    arrival_rate: float,
    service_rate_per_server: float,
    servers: int,
) -> float:
    if servers <= 0 or service_rate_per_server <= 0:
        raise ValueError("servers 和 service_rate_per_server 必须大于 0")

    capacity = servers * service_rate_per_server
    util = arrival_rate / capacity
    return float(np.clip(util, 0.0, 1.0))


def hourly_it_power_kw(
    arrival_rate: float,
    service_rate_per_server: float,
    servers: int,
    idle_power_kw: float,
    peak_power_kw: float,
) -> float:
    util = estimate_utilization(arrival_rate, service_rate_per_server, servers)
    per_server_power = idle_power_kw + util * (peak_power_kw - idle_power_kw)
    return servers * per_server_power


def hourly_total_power_kw(
    arrival_rate: float,
    service_rate_per_server: float,
    servers: int,
    idle_power_kw: float,
    peak_power_kw: float,
    pue: float,
) -> float:
    it_power = hourly_it_power_kw(
        arrival_rate=arrival_rate,
        service_rate_per_server=service_rate_per_server,
        servers=servers,
        idle_power_kw=idle_power_kw,
        peak_power_kw=peak_power_kw,
    )
    return it_power * pue


def total_energy_cost(
    arrival_rates,
    prices,
    server_plan,
    service_rate_per_server: float,
    idle_power_kw: float,
    peak_power_kw: float,
    pue: float,
    delta_t_hours: float,
) -> float:
    total_cost = 0.0
    for lam, price, servers in zip(arrival_rates, prices, server_plan):
        p_kw = hourly_total_power_kw(
            arrival_rate=float(lam),
            service_rate_per_server=service_rate_per_server,
            servers=int(servers),
            idle_power_kw=idle_power_kw,
            peak_power_kw=peak_power_kw,
            pue=pue,
        )
        total_cost += p_kw * delta_t_hours * float(price)
    return float(total_cost)


def total_carbon_emission(
    arrival_rates,
    carbon_factors,
    server_plan,
    service_rate_per_server: float,
    idle_power_kw: float,
    peak_power_kw: float,
    pue: float,
    delta_t_hours: float,
) -> float:
    total_carbon = 0.0
    for lam, cf, servers in zip(arrival_rates, carbon_factors, server_plan):
        p_kw = hourly_total_power_kw(
            arrival_rate=float(lam),
            service_rate_per_server=service_rate_per_server,
            servers=int(servers),
            idle_power_kw=idle_power_kw,
            peak_power_kw=peak_power_kw,
            pue=pue,
        )
        total_carbon += p_kw * delta_t_hours * float(cf)
    return float(total_carbon)