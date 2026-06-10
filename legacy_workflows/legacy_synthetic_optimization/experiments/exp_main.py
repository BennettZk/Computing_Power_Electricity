from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.exp_ablation import run_ablation_experiment
from experiments.exp_rolling import run_rolling_experiment
from experiments.exp_sensitivity import run_sensitivity_experiment
from models.objective import SchedulePlan, simulate_schedule
from optimizers.ga import run_ga
from optimizers.pso import run_pso
from schedulers.baseline_fcfs import build_fcfs_schedule
from schedulers.baseline_price_only import build_price_only_schedule
from schedulers.homogeneous_baseline import build_homogeneous_schedule
from schedulers.proposed_scheduler import run_proposed_scheduler
from utils.io_utils import ensure_inputs, ensure_output_dirs, load_all_configs, load_resource_pool, write_summary
from utils.metrics import RESULT_CSV_COLUMN_MAPPING, export_csv_chinese, summarize_result_rows
from utils.plotting import (
    plot_ablation_results,
    plot_convergence_curve,
    plot_cross_region_delay_sensitivity,
    plot_cpu_gpu_utilization,
    plot_load_shift_curve,
    plot_pareto_front,
    plot_power_breakdown_stack,
    plot_price_load_curve,
    plot_rolling_vs_static,
    plot_sensitivity_results,
    plot_single_objective_convergence,
    plot_spatial_migration_bar,
    plot_time_space_ablation,
    plot_time_space_migration_effect,
    plot_total_energy_bar,
)
from utils.seed import set_seed


def run_main_experiment() -> tuple[pd.DataFrame, dict]:
    """主实验入口：加载配置和数据，运行算法对比，导出表格与图表。"""
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
    ga_result = run_ga(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
    pso_result = run_pso(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
    proposed_result = run_proposed_scheduler(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)

    # 所有算法统一进入 simulate_schedule，保证指标口径一致。
    algorithm_runs = [
        ("FCFS", fcfs_schedule, "fcfs"),
        ("Price-Only", price_only_schedule, "price_only"),
        ("Homogeneous-Baseline", homogeneous_schedule, "fcfs"),
        ("GA", ga_result.best_schedule, "proposed"),
        ("PSO", pso_result.best_schedule, "proposed"),
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
    export_csv_chinese(results_df, results_csv_path, RESULT_CSV_COLUMN_MAPPING)

    # Proposed 的完整帕累托前沿单独保存，便于论文中绘制成本-时延散点图。
    pareto_csv_path = Path(experiment_cfg["paths"]["pareto_csv"])
    export_csv_chinese(proposed_result.pareto_df, pareto_csv_path, RESULT_CSV_COLUMN_MAPPING)

    # 导出论文实验常用图表。
    plot_price_load_curve(hourly_df, outputs_dir / "price_load_curve.png")
    plot_pareto_front(proposed_result.pareto_df, outputs_dir / "pareto_front.png")
    plot_total_energy_bar(results_df, outputs_dir / "energy_bar.png")
    plot_cpu_gpu_utilization(results_df, outputs_dir / "cpu_gpu_utilization.png")
    plot_convergence_curve(proposed_result.convergence_df, outputs_dir / "convergence_curve.png")
    plot_single_objective_convergence(ga_result.convergence_df, outputs_dir / "ga_convergence_curve.png", "GA Weighted Fitness Convergence")
    plot_single_objective_convergence(pso_result.convergence_df, outputs_dir / "pso_convergence_curve.png", "PSO Weighted Fitness Convergence")
    plot_spatial_migration_bar(results_df, outputs_dir / "spatial_migration_bar.png")
    plot_load_shift_curve(hourly_df, detailed_runs["Proposed"]["metrics"], outputs_dir / "load_shift_curve.png")

    proposed_metrics = detailed_runs["Proposed"]["metrics"]
    power_breakdown_df = pd.DataFrame(proposed_metrics.hourly_power_breakdown)
    export_csv_chinese(power_breakdown_df, outputs_dir / "hourly_power_breakdown.csv", RESULT_CSV_COLUMN_MAPPING)
    plot_power_breakdown_stack(power_breakdown_df, outputs_dir / "power_breakdown_stack.png")

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
    export_csv_chinese(ablation_df, ablation_csv_path, RESULT_CSV_COLUMN_MAPPING)
    plot_ablation_results(ablation_df, outputs_dir / "ablation_results.png")
    plot_time_space_ablation(ablation_df, outputs_dir / "time_space_ablation.png")

    best_schedule = proposed_result.best_schedule
    no_migration_schedule = SchedulePlan(
        cpu_servers=best_schedule.cpu_servers.copy(),
        gpu_servers=best_schedule.gpu_servers.copy(),
        defer_ratio=best_schedule.defer_ratio * 0.0,
        migration_ratio=best_schedule.migration_ratio * 0.0,
    )
    time_only_schedule = SchedulePlan(
        cpu_servers=best_schedule.cpu_servers.copy(),
        gpu_servers=best_schedule.gpu_servers.copy(),
        defer_ratio=best_schedule.defer_ratio.copy(),
        migration_ratio=best_schedule.migration_ratio * 0.0,
    )
    effect_rows: list[dict] = []
    for scenario, schedule in [
        ("无迁移", no_migration_schedule),
        ("仅时间迁移", time_only_schedule),
        ("时间+空间迁移", best_schedule),
    ]:
        metrics = simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="proposed",
        )
        effect_rows.append(
            {
                "scenario": scenario,
                "total_cost": metrics.total_cost,
                "avg_delay_hours": metrics.avg_delay_hours,
                "remote_task_count": metrics.remote_task_count,
                "sla_violation_rate": metrics.sla_violation_rate,
            }
        )
    plot_time_space_migration_effect(pd.DataFrame(effect_rows), outputs_dir / "time_space_migration_effect.png")

    rolling_result = run_rolling_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
        static_schedule=proposed_result.best_schedule,
    )
    rolling_csv_path = outputs_dir / "rolling_results.csv"
    export_csv_chinese(rolling_result.results_df, rolling_csv_path, RESULT_CSV_COLUMN_MAPPING)
    plot_rolling_vs_static(rolling_result.results_df, outputs_dir / "rolling_vs_static.png")

    sensitivity_df = run_sensitivity_experiment(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
    )
    sensitivity_csv_path = outputs_dir / "sensitivity_results.csv"
    export_csv_chinese(sensitivity_df, sensitivity_csv_path, RESULT_CSV_COLUMN_MAPPING)
    plot_sensitivity_results(sensitivity_df, outputs_dir / "sensitivity_results.png")
    plot_cross_region_delay_sensitivity(sensitivity_df, outputs_dir / "cross_region_delay_sensitivity.png")

    summary_lines = [
        "单数据中心异构资源调度实验摘要",
        "",
        f"任务总数: {len(tasks)}",
        f"电价场景: {price_cfg['scenario_name']}",
        f"Proposed 推荐方案总电费: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'total_cost'].iloc[0]):.4f}",
        f"Proposed 推荐方案平均时延: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'avg_delay_hours'].iloc[0]):.6f}",
        f"Proposed 推荐方案 SLA 违约率: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'sla_violation_rate'].iloc[0]):.6f}",
        f"Proposed 空间迁移任务数: {int(results_df.loc[results_df['algorithm'] == 'Proposed', 'remote_task_count'].iloc[0])}",
        f"Proposed 远端执行成本: {float(results_df.loc[results_df['algorithm'] == 'Proposed', 'remote_cost'].iloc[0]):.4f}",
        "",
        "主实验图表:",
        "outputs/price_load_curve.png",
        "outputs/pareto_front.png",
        "outputs/energy_bar.png",
        "outputs/cpu_gpu_utilization.png",
        "outputs/convergence_curve.png",
        "outputs/ga_convergence_curve.png",
        "outputs/pso_convergence_curve.png",
        "outputs/spatial_migration_bar.png",
        "outputs/load_shift_curve.png",
        "outputs/hourly_power_breakdown.csv",
        "outputs/power_breakdown_stack.png",
        "",
        "扩展实验输出:",
        "outputs/ablation_results.csv",
        "outputs/ablation_results.png",
        "outputs/time_space_ablation.png",
        "outputs/time_space_migration_effect.png",
        "outputs/rolling_results.csv",
        "outputs/rolling_vs_static.png",
        "outputs/sensitivity_results.csv",
        "outputs/sensitivity_results.png",
        "outputs/cross_region_delay_sensitivity.png",
    ]
    write_summary(experiment_cfg["paths"]["summary_txt"], summary_lines)

    return results_df, {
        "hourly_df": hourly_df,
        "tasks": tasks,
        "configs": configs,
        "resource_pool": resource_pool,
        "ga_result": ga_result,
        "pso_result": pso_result,
        "proposed_result": proposed_result,
        "rolling_result": rolling_result,
        "ablation_df": ablation_df,
        "sensitivity_df": sensitivity_df,
        "detailed_runs": detailed_runs,
    }


def main() -> None:
    # 脚本入口
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    run_main_experiment()


if __name__ == "__main__":
    main()
