from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.objective import simulate_schedule
from schedulers.baseline_fcfs import build_fcfs_schedule
from schedulers.baseline_price_only import build_price_only_schedule
from schedulers.homogeneous_baseline import build_homogeneous_schedule
from schedulers.proposed_scheduler import run_proposed_scheduler
from utils.io_utils import ensure_inputs, ensure_output_dirs, load_all_configs, load_resource_pool, write_summary
from utils.metrics import summarize_result_rows
from utils.plotting import (
    plot_convergence_curve,
    plot_cpu_gpu_utilization,
    plot_pareto_front,
    plot_price_load_curve,
    plot_total_energy_bar,
)
from utils.seed import set_seed


def run_main_experiment() -> tuple[pd.DataFrame, dict]:
    configs = load_all_configs()
    base_cfg = configs["base"]
    price_cfg = configs["price"]
    resource_pool = load_resource_pool(configs["resource"])
    experiment_cfg = configs["experiment"]

    set_seed(int(experiment_cfg["seed"]))
    hourly_df, tasks = ensure_inputs(configs)
    outputs_dir = ensure_output_dirs(experiment_cfg)

    fcfs_schedule = build_fcfs_schedule(tasks, resource_pool, base_cfg, experiment_cfg)
    price_only_schedule = build_price_only_schedule(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
    homogeneous_schedule = build_homogeneous_schedule(hourly_df, resource_pool, base_cfg, experiment_cfg)
    proposed_result = run_proposed_scheduler(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)

    algorithm_runs = [
        ("FCFS", fcfs_schedule, "fcfs"),
        ("Price-Only", price_only_schedule, "price_only"),
        ("Homogeneous-Baseline", homogeneous_schedule, "fcfs"),
        ("Proposed", proposed_result.best_schedule, "proposed"),
    ]

    result_rows: list[dict] = []
    detailed_runs: dict[str, dict] = {}
    for algorithm_name, schedule, mode in algorithm_runs:
        metrics = simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode=mode,
        )
        row = {"algorithm": algorithm_name, **metrics.to_dict()}
        result_rows.append(row)
        detailed_runs[algorithm_name] = {"schedule": schedule, "metrics": metrics}

    results_df = summarize_result_rows(result_rows)
    results_csv_path = Path(experiment_cfg["paths"]["results_csv"])
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

    pareto_csv_path = Path(experiment_cfg["paths"]["pareto_csv"])
    pareto_csv_path.parent.mkdir(parents=True, exist_ok=True)
    proposed_result.pareto_df.to_csv(pareto_csv_path, index=False, encoding="utf-8-sig")

    plot_price_load_curve(hourly_df, outputs_dir / "price_load_curve.png")
    plot_pareto_front(proposed_result.pareto_df, outputs_dir / "pareto_front.png")
    plot_total_energy_bar(results_df, outputs_dir / "energy_bar.png")
    plot_cpu_gpu_utilization(results_df, outputs_dir / "cpu_gpu_utilization.png")
    plot_convergence_curve(proposed_result.convergence_df, outputs_dir / "convergence_curve.png")

    summary_lines = [
        "Single data center heterogeneous scheduling experiment summary",
        "",
        f"Tasks: {len(tasks)}",
        f"Price scenario: {price_cfg['scenario_name']}",
        f"Best proposed total cost: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'total_cost'].iloc[0]):.4f}",
        f"Best proposed avg delay: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'avg_delay_hours'].iloc[0]):.6f}",
        f"Best proposed SLA violation rate: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'sla_violation_rate'].iloc[0]):.6f}",
        "",
        "Charts:",
        "outputs/price_load_curve.png",
        "outputs/pareto_front.png",
        "outputs/energy_bar.png",
        "outputs/cpu_gpu_utilization.png",
        "outputs/convergence_curve.png",
    ]
    write_summary(experiment_cfg["paths"]["summary_txt"], summary_lines)

    return results_df, {
        "hourly_df": hourly_df,
        "tasks": tasks,
        "proposed_result": proposed_result,
        "detailed_runs": detailed_runs,
    }


def main() -> None:
    run_main_experiment()


if __name__ == "__main__":
    main()
