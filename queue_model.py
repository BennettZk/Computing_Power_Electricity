import math


def _factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n 不能小于 0")
    return math.factorial(n)


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> float:
    """
    返回 Erlang-C 等待概率。
    arrival_rate: λ
    service_rate: μ
    servers: c
    """
    if servers <= 0:
        raise ValueError("servers 必须大于 0")
    if arrival_rate < 0 or service_rate <= 0:
        raise ValueError("arrival_rate >= 0 且 service_rate > 0")

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1.0

    a = arrival_rate / service_rate

    sum_terms = sum((a ** n) / _factorial(n) for n in range(servers))
    last_term = (a ** servers) / (_factorial(servers) * (1 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    pw = last_term * p0
    return max(0.0, min(1.0, pw))


def mmc_average_waiting_time_hours(
    arrival_rate: float,
    service_rate: float,
    servers: int,
) -> float:
    """
    M/M/c 平均排队等待时间 Wq（小时）
    """
    if arrival_rate == 0:
        return 0.0

    rho = arrival_rate / (servers * service_rate)
    if rho >= 1.0:
        return 1e4  # 系统爆掉时给大惩罚

    pw = erlang_c(arrival_rate, service_rate, servers)
    wq = pw / (servers * service_rate - arrival_rate)
    return max(0.0, wq)


def mmc_average_total_delay_hours(
    arrival_rate: float,
    service_rate: float,
    servers: int,
) -> float:
    """
    总时延 W = Wq + 1/μ
    """
    wq = mmc_average_waiting_time_hours(arrival_rate, service_rate, servers)
    service_time = 1.0 / service_rate if service_rate > 0 else 1e4
    return wq + service_time