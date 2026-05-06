from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.objective import SchedulePlan, simulate_schedule
from models.task import load_tasks_csv
from schedulers.baseline_fcfs import build_fcfs_schedule
from schedulers.baseline_price_only import build_price_only_schedule
from utils.io_utils import ensure_hourly_profile, load_all_configs, load_resource_pool


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _csv_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _record(
    rows: list[dict],
    tasks_path: Path,
    task_count: int,
    mode: str,
    step_name: str,
    elapsed_seconds: float,
    repeat_index: int,
) -> None:
    rows.append(
        {
            "scale_or_tasks_path": str(tasks_path),
            "task_count": task_count,
            "csv_size_mb": _csv_size_mb(tasks_path),
            "mode": mode,
            "step_name": step_name,
            "elapsed_seconds": elapsed_seconds,
            "repeat_index": repeat_index,
        }
    )


def _load_inputs(tasks_path: Path):
    configs = load_all_configs(PROJECT_ROOT / "config")
    hourly_df = ensure_hourly_profile(configs["base"], configs["price"])
    resource_pool = load_resource_pool(configs["resource"])
    tasks = load_tasks_csv(tasks_path)
    return configs, hourly_df, resource_pool, tasks


def _build_quick_proposed_schedule(price_only_schedule: SchedulePlan, hourly_df: pd.DataFrame, price_cfg: dict) -> SchedulePlan:
    high_price_threshold = float(hourly_df["price"].quantile(price_cfg["high_price_quantile"]))
    migration_ratio = np.where(hourly_df["price"].to_numpy(dtype=float) >= high_price_threshold, 0.2, 0.0)
    return SchedulePlan(
        cpu_servers=price_only_schedule.cpu_servers.copy(),
        gpu_servers=price_only_schedule.gpu_servers.copy(),
        defer_ratio=price_only_schedule.defer_ratio.copy(),
        migration_ratio=migration_ratio.astype(float),
    )


def run_quick_benchmark(tasks_path: Path, repeat: int) -> list[dict]:
    rows: list[dict] = []
    for repeat_index in range(1, repeat + 1):
        started = perf_counter()
        configs, hourly_df, resource_pool, tasks = _load_inputs(tasks_path)
        elapsed = perf_counter() - started
        task_count = len(tasks)
        base_cfg = configs["base"]
        price_cfg = configs["price"]
        experiment_cfg = configs["experiment"]
        _record(rows, tasks_path, task_count, "quick", "data_loading", elapsed, repeat_index)

        started = perf_counter()
        fcfs_schedule = build_fcfs_schedule(tasks, resource_pool, base_cfg, experiment_cfg)
        _record(rows, tasks_path, task_count, "quick", "build_fcfs_schedule", perf_counter() - started, repeat_index)

        started = perf_counter()
        simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=fcfs_schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="fcfs",
        )
        _record(rows, tasks_path, task_count, "quick", "simulate_fcfs", perf_counter() - started, repeat_index)

        started = perf_counter()
        price_only_schedule = build_price_only_schedule(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg)
        _record(rows, tasks_path, task_count, "quick", "build_price_only_schedule", perf_counter() - started, repeat_index)

        started = perf_counter()
        simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=price_only_schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="price_only",
        )
        _record(rows, tasks_path, task_count, "quick", "simulate_price_only", perf_counter() - started, repeat_index)

        started = perf_counter()
        quick_proposed_schedule = _build_quick_proposed_schedule(price_only_schedule, hourly_df, price_cfg)
        _record(rows, tasks_path, task_count, "quick", "build_quick_proposed_simplified_schedule", perf_counter() - started, repeat_index)

        started = perf_counter()
        simulate_schedule(
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            schedule=quick_proposed_schedule,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
            dispatch_mode="proposed",
        )
        _record(rows, tasks_path, task_count, "quick", "simulate_quick_proposed_simplified", perf_counter() - started, repeat_index)

    return rows


def run_full_benchmark(tasks_path: Path, repeat: int) -> list[dict]:
    rows: list[dict] = []
    task_count = _count_csv_rows(tasks_path)
    print("WARNING: full mode runs the complete main experiment and may take a long time on large task CSV files.")
    print("WARNING: full mode writes the normal experiment outputs such as results.csv, summary.txt, and outputs/*.csv/*.png.")

    import experiments.exp_main as exp_main

    original_load_all_configs = exp_main.load_all_configs

    def patched_load_all_configs():
        configs = original_load_all_configs()
        configs = {**configs}
        configs["experiment"] = {**configs["experiment"], "paths": {**configs["experiment"]["paths"]}}
        configs["experiment"]["paths"]["tasks_path"] = str(tasks_path)
        return configs

    for repeat_index in range(1, repeat + 1):
        exp_main.load_all_configs = patched_load_all_configs
        started = perf_counter()
        try:
            exp_main.run_main_experiment()
        finally:
            exp_main.load_all_configs = original_load_all_configs
        _record(rows, tasks_path, task_count, "full", "full_run_main_experiment", perf_counter() - started, repeat_index)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试不同规模任务数据下的调度仿真运行时间。")
    parser.add_argument("--tasks-path", default="data/synthetic/tasks.csv", help="任务 CSV 路径。")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="quick 只跑核心单次仿真；full 跑完整 main 实验。")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数。")
    parser.add_argument("--output", default="outputs/runtime_benchmark.csv", help="benchmark CSV 输出路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be a positive integer.")

    tasks_path = _project_path(args.tasks_path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Task CSV not found: {tasks_path}. Generate it first or pass --tasks-path.")

    rows = run_quick_benchmark(tasks_path, args.repeat) if args.mode == "quick" else run_full_benchmark(tasks_path, args.repeat)
    output_path = _project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Benchmark written to: {output_path}")
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
