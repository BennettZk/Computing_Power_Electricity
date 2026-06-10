from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objective import simulate_schedule
from models.task import load_tasks_csv
from optimizers.ga import run_ga
from optimizers.pso import run_pso
from schedulers.baseline_fcfs import build_fcfs_schedule
from schedulers.baseline_price_only import build_price_only_schedule
from schedulers.homogeneous_baseline import build_homogeneous_schedule
from schedulers.proposed_scheduler import run_proposed_scheduler
from utils.io_utils import load_resource_pool, load_yaml_like, write_summary
from utils.metrics import RESULT_CSV_COLUMN_MAPPING, export_csv_chinese, summarize_result_rows
from utils.plotting import plot_cpu_gpu_utilization, plot_spatial_migration_bar, plot_total_energy_bar
from utils.seed import set_seed


SCENARIO_TO_EXPERIMENT = {
    "5k": "experiment_real_5k.yaml",
    "10k": "experiment_real_10k.yaml",
    "full": "experiment_real_full.yaml",
}


def _load_real_configs(scenario: str, output_dir: str | None = None) -> dict[str, dict]:
    """加载真实场景实验所需的基础、资源和实验配置。"""
    config_dir = PROJECT_ROOT / "config" / "real_scenarios"
    experiment_cfg = load_yaml_like(config_dir / SCENARIO_TO_EXPERIMENT[scenario])
    price_cfg = load_yaml_like(PROJECT_ROOT / "config" / "price.yaml")
    price_cfg = {**price_cfg, "high_price_quantile": 0.45, "low_price_quantile": 0.25}
    if output_dir is not None:
        experiment_cfg = {**experiment_cfg, "paths": {**experiment_cfg["paths"]}}
        experiment_cfg["paths"]["outputs_dir"] = output_dir
        if "real_stress" in output_dir:
            experiment_cfg["paths"]["results_csv"] = f"{output_dir}/results_real_stress.csv"
            experiment_cfg["paths"]["summary_txt"] = f"{output_dir}/summary_real_stress.txt"
            experiment_cfg["paths"]["pareto_csv"] = f"{output_dir}/proposed_pareto_real_stress.csv"
        else:
            experiment_cfg["paths"]["results_csv"] = f"{output_dir}/results_real_quick.csv"
            experiment_cfg["paths"]["summary_txt"] = f"{output_dir}/summary_real_quick.txt"
            experiment_cfg["paths"]["pareto_csv"] = f"{output_dir}/proposed_pareto_real_{scenario}.csv"
    return {
        "base": load_yaml_like(config_dir / "base_real_scaled.yaml"),
        "resource": load_yaml_like(config_dir / "resource_real_scaled.yaml"),
        "experiment": experiment_cfg,
        "price": price_cfg,
    }


def _load_inputs(configs: dict[str, dict]) -> tuple[pd.DataFrame, list]:
    """加载真实场景实验的小时曲线和任务输入。"""
    experiment_cfg = configs["experiment"]
    tasks_path = PROJECT_ROOT / experiment_cfg["paths"]["tasks_path"]
    hourly_path = PROJECT_ROOT / experiment_cfg["paths"]["hourly_input_path"]
    if not tasks_path.exists() or not hourly_path.exists():
        raise FileNotFoundError(
            f"Missing real scaled inputs: {tasks_path} or {hourly_path}. "
            "Run python scripts/build_real_scenarios.py first."
        )
    hourly_df = pd.read_csv(hourly_path).sort_values("hour").reset_index(drop=True)
    required_hourly = {"hour", "arrival_rate", "price", "carbon_factor"}
    missing_hourly = required_hourly - set(hourly_df.columns)
    if missing_hourly:
        raise ValueError(f"Hourly input is missing columns: {sorted(missing_hourly)}")
    tasks = load_tasks_csv(tasks_path)
    return hourly_df, tasks


def _evaluate_run(name: str, schedule, mode: str, hourly_df, tasks, resource_pool, base_cfg, price_cfg) -> dict:
    """执行一次调度仿真并整理核心评价指标。"""
    metrics = simulate_schedule(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        schedule=schedule,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        dispatch_mode=mode,
    )
    return {"algorithm": name, **metrics.to_dict()}


def run_real_scenario(scenario: str = "5k", include_full_algorithms: bool = False, stress: bool = False) -> pd.DataFrame:
    """运行指定真实规模场景并导出结果。"""
    output_dir = "outputs/real_stress" if stress else "outputs/real_quick"
    configs = _load_real_configs(scenario, output_dir=output_dir)
    base_cfg = configs["base"]
    price_cfg = configs["price"]
    experiment_cfg = configs["experiment"]
    resource_pool = load_resource_pool(configs["resource"])
    set_seed(int(experiment_cfg["seed"]))
    hourly_df, tasks = _load_inputs(configs)

    outputs_dir = PROJECT_ROOT / experiment_cfg["paths"]["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    fcfs_schedule = build_fcfs_schedule(tasks, resource_pool, base_cfg, experiment_cfg)
    price_only_schedule = build_price_only_schedule(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
    proposed_result = run_proposed_scheduler(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)

    runs = [
        ("FCFS", fcfs_schedule, "fcfs"),
        ("Price-Only", price_only_schedule, "price_only"),
        ("Proposed", proposed_result.best_schedule, "proposed"),
    ]
    if include_full_algorithms:
        homogeneous_schedule = build_homogeneous_schedule(hourly_df, resource_pool, base_cfg, experiment_cfg)
        ga_result = run_ga(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
        pso_result = run_pso(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
        runs.extend(
            [
                ("Homogeneous-Baseline", homogeneous_schedule, "fcfs"),
                ("GA", ga_result.best_schedule, "proposed"),
                ("PSO", pso_result.best_schedule, "proposed"),
            ]
        )

    rows = [_evaluate_run(name, schedule, mode, hourly_df, tasks, resource_pool, base_cfg, price_cfg) for name, schedule, mode in runs]
    results_df = summarize_result_rows(rows)
    elapsed = perf_counter() - started

    results_path = PROJECT_ROOT / experiment_cfg["paths"]["results_csv"]
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    cn_path = results_path.with_name(results_path.stem + "_cn.csv")
    export_csv_chinese(results_df, cn_path, RESULT_CSV_COLUMN_MAPPING)
    proposed_result.pareto_df.to_csv(PROJECT_ROOT / experiment_cfg["paths"]["pareto_csv"], index=False, encoding="utf-8-sig")

    plot_total_energy_bar(results_df, outputs_dir / "energy_bar.png")
    plot_cpu_gpu_utilization(results_df, outputs_dir / "cpu_gpu_utilization.png")
    plot_spatial_migration_bar(results_df, outputs_dir / "spatial_migration_bar.png")

    proposed_row = results_df.loc[results_df["algorithm"] == "Proposed"].iloc[0]
    summary_lines = [
        "真实缩放场景快速运行摘要" if not stress else "真实全量压力测试摘要",
        "",
        f"场景: real_{scenario}",
        f"任务总数: {len(tasks)}",
        f"运行耗时/s: {elapsed:.3f}",
        f"Proposed 完成任务数: {int(proposed_row['total_completed_tasks'])}",
        f"Proposed 任务完成率: {float(proposed_row['completion_rate']):.6f}",
        f"Proposed SLA 违约率: {float(proposed_row['sla_violation_rate']):.6f}",
        f"Proposed 平均时延/h: {float(proposed_row['avg_delay_hours']):.6f}",
        f"Proposed 远端迁移任务数: {int(proposed_row['remote_task_count'])}",
        f"Proposed GPU平均利用率: {float(proposed_row['avg_gpu_utilization']):.6f}",
        f"Proposed Token总量: {float(proposed_row['total_tokens']):.3f}",
        "",
        f"英文结果CSV: {results_path}",
        f"中文展示CSV: {cn_path}",
    ]
    if stress:
        summary_lines.extend(
            [
                "",
                "提示: 原始真实数据规模远超当前单数据中心等效资源规模，该结果用于压力测试和模型适用边界分析，不作为主实验算法优劣结论。",
            ]
        )
    write_summary(PROJECT_ROOT / experiment_cfg["paths"]["summary_txt"], summary_lines)
    return results_df


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回脚本运行配置。"""
    parser = argparse.ArgumentParser(description="运行真实缩放场景 quick/stress 实验，不覆盖默认主实验输出。")
    parser.add_argument("--scenario", choices=["5k", "10k", "full"], default="5k")
    parser.add_argument("--quick", action="store_true", help="保留兼容参数；默认就是 quick。")
    parser.add_argument("--full", action="store_true", help="额外运行 Homogeneous、GA、PSO；不运行消融、滚动和灵敏度。")
    parser.add_argument("--stress", action="store_true", help="以压力测试口径输出到 outputs/real_stress/。")
    return parser.parse_args()


def main() -> None:
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    args = parse_args()
    scenario = "full" if args.stress else args.scenario
    run_real_scenario(scenario=scenario, include_full_algorithms=args.full, stress=args.stress)


if __name__ == "__main__":
    main()
