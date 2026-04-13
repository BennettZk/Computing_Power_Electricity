from __future__ import annotations

import math


def _factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative.")
    return math.factorial(n)


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> float:
    if servers <= 0:
        return 1.0
    if arrival_rate < 0 or service_rate <= 0:
        raise ValueError("arrival_rate must be >= 0 and service_rate must be > 0.")

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1.0

    a = arrival_rate / service_rate
    sum_terms = sum((a ** n) / _factorial(n) for n in range(servers))
    last_term = (a ** servers) / (_factorial(servers) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    return max(0.0, min(1.0, last_term * p0))


def mmc_average_waiting_time_hours(arrival_rate: float, service_rate: float, servers: int) -> float:
    if arrival_rate == 0:
        return 0.0
    if servers <= 0 or service_rate <= 0:
        return 1e4

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1e4

    return erlang_c(arrival_rate, service_rate, servers) / (servers * service_rate - arrival_rate)


def mmc_average_total_delay_hours(arrival_rate: float, service_rate: float, servers: int) -> float:
    queue_wait = mmc_average_waiting_time_hours(arrival_rate, service_rate, servers)
    service_time = 1.0 / service_rate if service_rate > 0 else 1e4
    return queue_wait + service_time


def estimate_delay_statistics(
    arrival_rate: float,
    service_rate: float,
    servers: int,
    waited_slots: float,
    slot_hours: float,
) -> dict[str, float]:
    within_slot_delay = mmc_average_total_delay_hours(arrival_rate, service_rate, servers)
    total_delay = max(0.0, waited_slots) * slot_hours + within_slot_delay
    return {
        "within_slot_delay_hours": within_slot_delay,
        "total_delay_hours": total_delay,
    }
