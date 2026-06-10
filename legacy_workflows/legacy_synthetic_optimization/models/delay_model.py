from __future__ import annotations

import math


def _factorial(n: int) -> int:
    """计算非负整数阶乘，并对负数输入给出错误。"""
    if n < 0:
        raise ValueError("n must be non-negative.")
    return math.factorial(n)


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Erlang-C 等待概率，用于近似 M/M/c 队列的排队延迟。"""
    if servers <= 0:
        return 1.0
    if arrival_rate < 0 or service_rate <= 0:
        raise ValueError("arrival_rate must be >= 0 and service_rate must be > 0.")

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1.0

    a = arrival_rate / service_rate
    if a == 0.0:
        return 0.0

    # 使用 log-sum-exp，避免真实规模场景中服务器数量过多导致数值溢出。
    log_terms = [n * math.log(a) - math.lgamma(n + 1) for n in range(servers)]
    log_last = servers * math.log(a) - math.lgamma(servers + 1) - math.log(1.0 - rho)
    max_log = max(max(log_terms), log_last)
    scaled_sum = sum(math.exp(value - max_log) for value in log_terms)
    scaled_last = math.exp(log_last - max_log)
    probability = scaled_last / (scaled_sum + scaled_last)
    return max(0.0, min(1.0, probability))


def mmc_average_waiting_time_hours(arrival_rate: float, service_rate: float, servers: int) -> float:
    """返回 M/M/c 平均排队等待时间，系统不稳定时给一个大惩罚值。"""
    if arrival_rate == 0:
        return 0.0
    if servers <= 0 or service_rate <= 0:
        return 1e4

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1e4

    return erlang_c(arrival_rate, service_rate, servers) / (servers * service_rate - arrival_rate)


def mmc_average_total_delay_hours(arrival_rate: float, service_rate: float, servers: int) -> float:
    """总时延 = 排队等待时间 + 平均服务时间。"""
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
    """统一输出时隙内排队时延和包含跨时隙等待的总时延。"""
    within_slot_delay = mmc_average_total_delay_hours(arrival_rate, service_rate, servers)
    total_delay = max(0.0, waited_slots) * slot_hours + within_slot_delay
    return {
        "within_slot_delay_hours": within_slot_delay,
        "total_delay_hours": total_delay,
    }
