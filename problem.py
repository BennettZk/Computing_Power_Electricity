import numpy as np
from pymoo.core.problem import ElementwiseProblem

from config import SystemConfig
from queue_model import mmc_average_total_delay_hours
from power_model import total_energy_cost, total_carbon_emission


class DataCenterSchedulingProblem(ElementwiseProblem):
    def __init__(self, hourly_df, cfg: SystemConfig):
        self.df = hourly_df
        self.cfg = cfg

        super().__init__(
            n_var=cfg.hours,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.full(cfg.hours, cfg.min_servers),
            xu=np.full(cfg.hours, cfg.max_servers),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        server_plan = np.rint(x).astype(int)

        arrival_rates = self.df["arrival_rate"].to_numpy(dtype=float)
        prices = self.df["price"].to_numpy(dtype=float)
        carbon_factors = self.df["carbon_factor"].to_numpy(dtype=float)

        # 1. 总电费
        total_cost = total_energy_cost(
            arrival_rates=arrival_rates,
            prices=prices,
            server_plan=server_plan,
            service_rate_per_server=self.cfg.service_rate_per_server,
            idle_power_kw=self.cfg.server_idle_power_kw,
            peak_power_kw=self.cfg.server_peak_power_kw,
            pue=self.cfg.pue,
            delta_t_hours=self.cfg.delta_t_hours,
        )

        # 2. 平均时延
        delays = []
        infeasible_count = 0

        for lam, servers in zip(arrival_rates, server_plan):
            delay = mmc_average_total_delay_hours(
                arrival_rate=float(lam),
                service_rate=self.cfg.service_rate_per_server,
                servers=int(servers),
            )
            delays.append(delay)
            if delay >= 1e4:
                infeasible_count += 1

        avg_delay = float(np.mean(delays))

        # 如果有爆系统的时段，加大惩罚
        if infeasible_count > 0:
            total_cost += self.cfg.infeasible_penalty * infeasible_count
            avg_delay += self.cfg.infeasible_penalty * infeasible_count

        # 约束：平均时延不能超过阈值
        g1 = avg_delay - self.cfg.max_avg_delay_hours

        out["F"] = [total_cost, avg_delay]
        out["G"] = [g1]

    def evaluate_metrics(self, server_plan: np.ndarray) -> dict:
        arrival_rates = self.df["arrival_rate"].to_numpy(dtype=float)
        prices = self.df["price"].to_numpy(dtype=float)
        carbon_factors = self.df["carbon_factor"].to_numpy(dtype=float)

        total_cost = total_energy_cost(
            arrival_rates=arrival_rates,
            prices=prices,
            server_plan=server_plan,
            service_rate_per_server=self.cfg.service_rate_per_server,
            idle_power_kw=self.cfg.server_idle_power_kw,
            peak_power_kw=self.cfg.server_peak_power_kw,
            pue=self.cfg.pue,
            delta_t_hours=self.cfg.delta_t_hours,
        )

        total_carbon = total_carbon_emission(
            arrival_rates=arrival_rates,
            carbon_factors=carbon_factors,
            server_plan=server_plan,
            service_rate_per_server=self.cfg.service_rate_per_server,
            idle_power_kw=self.cfg.server_idle_power_kw,
            peak_power_kw=self.cfg.server_peak_power_kw,
            pue=self.cfg.pue,
            delta_t_hours=self.cfg.delta_t_hours,
        )

        delays = [
            mmc_average_total_delay_hours(
                arrival_rate=float(lam),
                service_rate=self.cfg.service_rate_per_server,
                servers=int(s),
            )
            for lam, s in zip(arrival_rates, server_plan)
        ]
        avg_delay = float(np.mean(delays))

        processed_requests = float(np.sum(arrival_rates))
        total_tokens = processed_requests * self.cfg.avg_tokens_per_request

        cost_per_1k_tokens = total_cost / (total_tokens / 1000.0) if total_tokens > 0 else np.inf
        carbon_per_1k_tokens = total_carbon / (total_tokens / 1000.0) if total_tokens > 0 else np.inf

        return {
            "total_cost": total_cost,
            "avg_delay_hours": avg_delay,
            "total_carbon": total_carbon,
            "total_tokens": total_tokens,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "carbon_per_1k_tokens": carbon_per_1k_tokens,
            "hourly_delays": delays,
        }