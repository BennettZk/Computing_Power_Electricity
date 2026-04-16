from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from models.objective import SchedulePlan, simulate_schedule
from models.resource import ResourcePool
from models.task import Task


@dataclass
class NSGA2RunResult:
    pareto_df: pd.DataFrame
    best_schedule: SchedulePlan
    best_metrics: dict
    convergence_df: pd.DataFrame


class HeterogeneousSchedulingProblem(ElementwiseProblem):
    def __init__(
        self,
        hourly_df: pd.DataFrame,
        tasks: list[Task],
        resource_pool: ResourcePool,
        base_cfg: dict,
        price_cfg: dict,
        experiment_cfg: dict,
    ) -> None:
        self.hourly_df = hourly_df
        self.tasks = tasks
        self.resource_pool = resource_pool
        self.base_cfg = base_cfg
        self.price_cfg = price_cfg
        self.experiment_cfg = experiment_cfg

        hours = int(base_cfg["time"]["hours"])
        defer_levels = int(experiment_cfg["optimizer"]["defer_ratio_levels"])
        xl = np.concatenate(
            [
                np.full(hours, int(experiment_cfg["optimizer"]["cpu_min_active"])),
                np.full(hours, int(experiment_cfg["optimizer"]["gpu_min_active"])),
                np.zeros(hours, dtype=int),
            ]
        )
        xu = np.concatenate(
            [
                np.full(hours, resource_pool.cpu.count),
                np.full(hours, resource_pool.gpu.count),
                np.full(hours, defer_levels),
            ]
        )

        super().__init__(
            n_var=hours * 3,
            n_obj=3,
            n_ieq_constr=3,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def decode(self, x: np.ndarray) -> SchedulePlan:
        hours = int(self.base_cfg["time"]["hours"])
        defer_levels = max(1, int(self.experiment_cfg["optimizer"]["defer_ratio_levels"]))
        cpu = np.rint(x[:hours]).astype(int)
        gpu = np.rint(x[hours : 2 * hours]).astype(int)
        defer = np.rint(x[2 * hours :]).astype(int) / defer_levels
        return SchedulePlan(cpu_servers=cpu, gpu_servers=gpu, defer_ratio=defer)

    def _evaluate(self, x, out, *args, **kwargs):
        schedule = self.decode(np.asarray(x, dtype=float))
        metrics = simulate_schedule(
            hourly_df=self.hourly_df,
            tasks=self.tasks,
            resource_pool=self.resource_pool,
            schedule=schedule,
            base_cfg=self.base_cfg,
            price_cfg=self.price_cfg,
            dispatch_mode="proposed",
        )

        out["F"] = [
            metrics.total_cost + metrics.objective_penalty,
            metrics.avg_delay_hours + metrics.sla_violation_rate * 5.0 + metrics.objective_penalty * 1e-4,
            metrics.load_imbalance + metrics.delay_sensitive_violation_rate * 2.0,
        ]
        out["G"] = [
            metrics.avg_delay_hours - float(self.base_cfg["constraints"]["max_avg_delay_hours"]),
            metrics.sla_violation_rate - float(self.base_cfg["constraints"]["max_sla_violation_rate"]),
            float(metrics.power_limit_violation_hours + metrics.peak_limit_violation_hours),
        ]


def run_nsga2(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
) -> NSGA2RunResult:
    problem = HeterogeneousSchedulingProblem(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
    )

    optimizer_cfg = experiment_cfg["optimizer"]
    algorithm = NSGA2(
        pop_size=int(optimizer_cfg["pop_size"]),
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", int(optimizer_cfg["n_gen"])),
        seed=int(experiment_cfg["seed"]),
        save_history=True,
        verbose=False,
    )

    raw_x = result.X
    if raw_x is None:
        raw_x = result.pop.get("X")
    X = np.atleast_2d(raw_x)
    rows: list[dict] = []
    for idx, solution in enumerate(X):
        schedule = problem.decode(solution)
        metrics = simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="proposed",
        )
        rows.append(
            {
                "solution_id": idx,
                "total_cost": metrics.total_cost,
                "avg_delay_hours": metrics.avg_delay_hours,
                "load_imbalance": metrics.load_imbalance,
                "sla_violation_rate": metrics.sla_violation_rate,
                "cpu_servers": ",".join(map(str, schedule.cpu_servers.tolist())),
                "gpu_servers": ",".join(map(str, schedule.gpu_servers.tolist())),
                "defer_ratio": ",".join(f"{value:.1f}" for value in schedule.defer_ratio.tolist()),
                **metrics.to_dict(),
            }
        )

    pareto_df = pd.DataFrame(rows).sort_values(["total_cost", "avg_delay_hours", "load_imbalance"]).reset_index(drop=True)
    feasible_df = pareto_df[
        (pareto_df["avg_delay_hours"] <= float(base_cfg["constraints"]["max_avg_delay_hours"]))
        & (pareto_df["sla_violation_rate"] <= float(base_cfg["constraints"]["max_sla_violation_rate"]))
        & (pareto_df["power_limit_violation_hours"] <= 0)
        & (pareto_df["peak_limit_violation_hours"] <= 0)
    ]
    recommendation_df = feasible_df if not feasible_df.empty else pareto_df
    recommended = recommendation_df.sort_values(["total_cost", "avg_delay_hours", "load_imbalance"]).iloc[0]

    best_schedule = SchedulePlan(
        cpu_servers=np.array([int(value) for value in str(recommended["cpu_servers"]).split(",")], dtype=int),
        gpu_servers=np.array([int(value) for value in str(recommended["gpu_servers"]).split(",")], dtype=int),
        defer_ratio=np.array([float(value) for value in str(recommended["defer_ratio"]).split(",")], dtype=float),
    )

    convergence_rows: list[dict] = []
    for generation, history_item in enumerate(result.history):
        f_values = np.atleast_2d(history_item.pop.get("F"))
        convergence_rows.append(
            {
                "generation": generation,
                "best_total_cost": float(np.min(f_values[:, 0])),
                "best_delay_objective": float(np.min(f_values[:, 1])),
                "best_balance_objective": float(np.min(f_values[:, 2])),
            }
        )

    return NSGA2RunResult(
        pareto_df=pareto_df,
        best_schedule=best_schedule,
        best_metrics=recommended.to_dict(),
        convergence_df=pd.DataFrame(convergence_rows),
    )
