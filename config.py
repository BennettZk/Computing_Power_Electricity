from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    # 时间设置
    hours: int = 24
    delta_t_hours: float = 1.0

    # 服务器设置
    min_servers: int = 1
    max_servers: int = 20

    # 单台服务器服务能力（每小时最多处理多少请求）
    service_rate_per_server: float = 25.0

    # 功率模型（kW）
    server_idle_power_kw: float = 0.18
    server_peak_power_kw: float = 0.42

    # PUE 简化处理
    pue: float = 1.25

    # 每个请求平均产出 token 数
    avg_tokens_per_request: float = 800.0

    # 约束和惩罚
    max_avg_delay_hours: float = 2.0
    infeasible_penalty: float = 1e6