from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.exp_ablation import run_ablation_experiment
from experiments.exp_sensitivity import run_sensitivity_experiment
from models.objective import simulate_schedule
from schedulers.baseline_fcfs import build_fcfs_schedule
from schedulers.baseline_price_only import build_price_only_schedule
from schedulers.homogeneous_baseline import build_homogeneous_schedule
from schedulers.proposed_scheduler import run_proposed_scheduler
from utils.io_utils import ensure_inputs, ensure_output_dirs, load_all_configs, load_resource_pool, write_summary
from utils.metrics import summarize_result_rows
from utils.plotting import (
    plot_ablation_results,
    plot_convergence_curve,
    plot_cpu_gpu_utilization,
    plot_pareto_front,
    plot_price_load_curve,
    plot_sensitivity_results,
    plot_total_energy_bar,
)
from utils.seed import set_seed


def run_main_experiment() -> tuple[pd.DataFrame, dict]:
    """主实验入口：加载配置和数据，运行四种算法，导出表格与图表。"""
    configs = load_all_configs()
    base_cfg = configs["base"]
    price_cfg = configs["price"]
    resource_pool = load_resource_pool(configs["resource"])
    experiment_cfg = configs["experiment"]

    set_seed(int(experiment_cfg["seed"]))
    # 输入不存在时会自动生成默认小时曲线和合成任务，保证首次运行可复现。
    hourly_df, tasks = ensure_inputs(configs)
    outputs_dir = ensure_output_dirs(experiment_cfg)

    fcfs_schedule = build_fcfs_schedule(tasks, resource_pool, base_cfg, experiment_cfg)
    price_only_schedule = build_price_only_schedule(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
    homogeneous_schedule = build_homogeneous_schedule(hourly_df, resource_pool, base_cfg, experiment_cfg)
    proposed_result = run_proposed_scheduler(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)

    # 所有算法统一进入 simulate_schedule，保证指标口径一致。
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

    # Proposed 的完整帕累托前沿单独保存，便于论文中绘制成本-时延散点图。
    pareto_csv_path = Path(experiment_cfg["paths"]["pareto_csv"])
    pareto_csv_path.parent.mkdir(parents=True, exist_ok=True)
    proposed_result.pareto_df.to_csv(pareto_csv_path, index=False, encoding="utf-8-sig")

    # 自动导出论文实验常用图表。
    plot_price_load_curve(hourly_df, outputs_dir / "price_load_curve.png")
    plot_pareto_front(proposed_result.pareto_df, outputs_dir / "pareto_front.png")
    plot_total_energy_bar(results_df, outputs_dir / "energy_bar.png")
    plot_cpu_gpu_utilization(results_df, outputs_dir / "cpu_gpu_utilization.png")
    plot_convergence_curve(proposed_result.convergence_df, outputs_dir / "convergence_curve.png")

    ablation_df = run_ablation_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
        proposed_schedule=proposed_result.best_schedule,
        homogeneous_schedule=homogeneous_schedule,
    )
    ablation_csv_path = outputs_dir / "ablation_results.csv"
    ablation_df.to_csv(ablation_csv_path, index=False, encoding="utf-8-sig")
    plot_ablation_results(ablation_df, outputs_dir / "ablation_results.png")

    sensitivity_df = run_sensitivity_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
    )
    sensitivity_csv_path = outputs_dir / "sensitivity_results.csv"
    sensitivity_df.to_csv(sensitivity_csv_path, index=False, encoding="utf-8-sig")
    plot_sensitivity_results(sensitivity_df, outputs_dir / "sensitivity_results.png")

    summary_lines = [
        "单数据中心异构资源调度实验摘要",
        "",
        f"任务总数: {len(tasks)}",
        f"电价场景: {price_cfg['scenario_name']}",
        f"Proposed 推荐方案总电费: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'total_cost'].iloc[0]):.4f}",
        f"Proposed 推荐方案平均时延: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'avg_delay_hours'].iloc[0]):.6f}",
        f"Proposed 推荐方案 SLA 违约率: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'sla_violation_rate'].iloc[0]):.6f}",
        "",
        "主实验图表:",
        "outputs/price_load_curve.png",
        "outputs/pareto_front.png",
        "outputs/energy_bar.png",
        "outputs/cpu_gpu_utilization.png",
        "outputs/convergence_curve.png",
        "",
        "扩展实验输出:",
        "outputs/ablation_results.csv",
        "outputs/ablation_results.png",
        "outputs/sensitivity_results.csv",
        "outputs/sensitivity_results.png",
    ]
    write_summary(experiment_cfg["paths"]["summary_txt"], summary_lines)

    return results_df, {
        "hourly_df": hourly_df,
        "tasks": tasks,
        "configs": configs,
        "resource_pool": resource_pool,
        "proposed_result": proposed_result,
        "ablation_df": ablation_df,
        "sensitivity_df": sensitivity_df,
        "detailed_runs": detailed_runs,
    }


def main() -> None:
    run_main_experiment()


if __name__ == "__main__":
    main()
