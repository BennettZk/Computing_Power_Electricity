import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from config import SystemConfig
from data_loader import load_hourly_data
from problem import DataCenterSchedulingProblem


def main():
    cfg = SystemConfig()
    df = load_hourly_data("data/hourly_input.csv")
    problem = DataCenterSchedulingProblem(df, cfg)

    algorithm = NSGA2(
        pop_size=80,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", 100),
        seed=42,
        save_history=False,
        verbose=True,
    )

    X = np.rint(result.X).astype(int)
    F = result.F

    rows = []
    for i in range(len(X)):
        metrics = problem.evaluate_metrics(X[i])
        row = {
            "solution_id": i,
            "total_cost": metrics["total_cost"],
            "avg_delay_hours": metrics["avg_delay_hours"],
            "total_tokens": metrics["total_tokens"],
            "cost_per_1k_tokens": metrics["cost_per_1k_tokens"],
            "carbon_per_1k_tokens": metrics["carbon_per_1k_tokens"],
            "server_plan": ",".join(map(str, X[i].tolist())),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["total_cost", "avg_delay_hours"])
    out_df.to_csv("nsga2_results.csv", index=False, encoding="utf-8-sig")

    # 取一个折中解：目标归一化后总和最小
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    F_norm = (F - f_min) / np.maximum(f_max - f_min, 1e-9)
    best_idx = np.argmin(F_norm.sum(axis=1))

    best_plan = X[best_idx]
    metrics = problem.evaluate_metrics(best_plan)

    print("\n=== 推荐折中解 ===")
    print("solution_id:", best_idx)
    print("server_plan:", best_plan.tolist())
    print("total_cost:", round(metrics["total_cost"], 4))
    print("avg_delay_hours:", round(metrics["avg_delay_hours"], 6))
    print("total_tokens:", round(metrics["total_tokens"], 2))
    print("cost_per_1k_tokens:", round(metrics["cost_per_1k_tokens"], 6))
    print("carbon_per_1k_tokens:", round(metrics["carbon_per_1k_tokens"], 6))


if __name__ == "__main__":
    main()