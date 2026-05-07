from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from models.delay_model import mmc_average_total_delay_hours
from models.objective import SchedulePlan
from models.power_model import total_energy_cost
from models.resource import ResourcePool


def _fallback_total_server_plan(hourly_df: pd.DataFrame, resource_pool: ResourcePool, base_cfg: dict) -> np.ndarray:
    """当同构 NSGA-II 没有返回可用种群时，给出保守的启发式总服务器计划。"""
    hours = int(base_cfg["time"]["hours"])
    total_servers = resource_pool.cpu.count + resource_pool.gpu.count
    effective_service = (resource_pool.cpu.queue_service_rate + resource_pool.gpu.queue_service_rate) / 2.0
    arrival = hourly_df["arrival_rate"].to_numpy(dtype=float)
    estimated = np.ceil(arrival / max(effective_service, 1e-9)).astype(int)
    return np.clip(estimated, 1, total_servers)[:hours]


class HomogeneousSchedulingProblem(ElementwiseProblem):
    """同构基线问题：沿用原论文“单一服务器类型 + 单一负载”的建模思路。"""

    def __init__(self, hourly_df: pd.DataFrame, resource_pool: ResourcePool, base_cfg: dict):
        self.hourly_df = hourly_df
        self.resource_pool = resource_pool
        self.base_cfg = base_cfg
        hours = int(base_cfg["time"]["hours"])
        total_servers = resource_pool.cpu.count + resource_pool.gpu.count
        super().__init__(
            n_var=hours,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.ones(hours, dtype=int),
            xu=np.full(hours, total_servers, dtype=int),
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        server_plan = np.rint(x).astype(int)
        price = self.hourly_df["price"].to_numpy(dtype=float)
        arrival = self.hourly_df["arrival_rate"].to_numpy(dtype=float)

        # 将 CPU/GPU 参数折算成一个等效服务器，用于构造同构对照组。
        effective_service = (self.resource_pool.cpu.queue_service_rate + self.resource_pool.gpu.queue_service_rate) / 2.0
        effective_idle = (self.resource_pool.cpu.idle_power + self.resource_pool.gpu.idle_power) / 2.0
        effective_peak = (self.resource_pool.cpu.peak_power + self.resource_pool.gpu.peak_power) / 2.0

        total_cost = total_energy_cost(
            arrival_rates=arrival,
            prices=price,
            server_plan=server_plan,
            service_rate_per_server=effective_service,
            idle_power_kw=effective_idle,
            peak_power_kw=effective_peak,
            pue=1.2,
            delta_t_hours=float(self.base_cfg["time"]["slot_hours"]),
        )
        delays = [mmc_average_total_delay_hours(float(lam), effective_service, int(servers)) for lam, servers in zip(arrival, server_plan)]
        avg_delay = float(np.mean(delays))

        out["F"] = [total_cost, avg_delay]
        out["G"] = [avg_delay - float(self.base_cfg["constraints"]["max_avg_delay_hours"])]


def build_homogeneous_schedule(hourly_df: pd.DataFrame, resource_pool: ResourcePool, base_cfg: dict, experiment_cfg: dict) -> SchedulePlan:
    """运行同构 NSGA-II，并把总服务器台数按 CPU/GPU 数量比例映射回异构仿真接口。"""
    problem = HomogeneousSchedulingProblem(hourly_df, resource_pool, base_cfg)
    algorithm = NSGA2(
        pop_size=max(20, int(experiment_cfg["optimizer"]["pop_size"]) // 2),
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", max(10, int(experiment_cfg["optimizer"]["n_gen"]) // 2)),
        seed=int(experiment_cfg["seed"]),
        save_history=False,
        verbose=False,
    )

    raw_x = result.X
    raw_f = result.F
    if raw_x is None or raw_f is None:
        # 真实大负载下同构约束可能完全不可行，此时 pymoo 不返回 result.X/F。
        # 回退到最终种群继续选择折中解，保证对照组不让主实验中断。
        raw_x = result.pop.get("X") if result.pop is not None else None
        raw_f = result.pop.get("F") if result.pop is not None else None

    if raw_x is None or raw_f is None:
        total_servers = _fallback_total_server_plan(hourly_df, resource_pool, base_cfg)
    else:
        X = np.atleast_2d(raw_x)
        F = np.atleast_2d(raw_f)
        norm = (F - F.min(axis=0)) / np.maximum(F.max(axis=0) - F.min(axis=0), 1e-9)
        # 归一化后取成本和时延综合最小的折中解。
        best_idx = int(np.argmin(norm.sum(axis=1)))
        total_servers = np.rint(X[best_idx]).astype(int)

    cpu_share = resource_pool.cpu.count / max(resource_pool.cpu.count + resource_pool.gpu.count, 1)
    cpu_servers = np.clip(np.ceil(total_servers * cpu_share), 1, resource_pool.cpu.count).astype(int)
    gpu_servers = np.clip(total_servers - cpu_servers, 0, resource_pool.gpu.count).astype(int)

    return SchedulePlan(
        cpu_servers=cpu_servers,
        gpu_servers=gpu_servers,
        defer_ratio=np.zeros(int(base_cfg["time"]["hours"]), dtype=float),
        migration_ratio=np.zeros(int(base_cfg["time"]["hours"]), dtype=float),
    )
