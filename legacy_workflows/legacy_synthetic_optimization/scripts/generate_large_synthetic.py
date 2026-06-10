from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.task import generate_synthetic_tasks
from utils.io_utils import ensure_hourly_profile, load_all_configs


REPORT_PATH = PROJECT_ROOT / "outputs" / "large_synthetic_report.txt"


def _project_path(path: str | Path) -> Path:
    """将输入路径解析为项目根目录下的绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _count_csv_rows(path: Path) -> int:
    """统计 CSV 文件的数据行数。"""
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _load_hourly_profile(configs: dict[str, dict]) -> pd.DataFrame:
    """加载生成大规模合成数据所需的小时曲线。"""
    hourly_df = ensure_hourly_profile(configs["base"], configs["price"])
    required_cols = {"hour", "arrival_rate", "price", "carbon_factor"}
    missing = required_cols - set(hourly_df.columns)
    if missing:
        raise ValueError(f"Hourly profile is missing columns: {sorted(missing)}")
    if len(hourly_df) != 24:
        raise ValueError("The large synthetic generator expects exactly 24 hourly rows.")
    return hourly_df.sort_values("hour").reset_index(drop=True)


def _hourly_for_scale(hourly_df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """按负载倍率生成小时级输入曲线。"""
    if scale <= 0:
        raise ValueError("--scale must be greater than 0.")
    scaled = hourly_df.copy()
    counts = np.rint(scaled["arrival_rate"].to_numpy(dtype=float) * scale).astype(int)
    scaled["arrival_rate"] = np.clip(counts, 0, None)
    if int(scaled["arrival_rate"].sum()) <= 0:
        raise ValueError("Scaled hourly arrival rates produce zero tasks.")
    return scaled


def _hourly_for_target_tasks(hourly_df: pd.DataFrame, target_tasks: int) -> pd.DataFrame:
    """按目标任务数生成小时级输入曲线。"""
    if target_tasks <= 0:
        raise ValueError("--target-tasks must be a positive integer.")

    targeted = hourly_df.copy()
    base_counts = targeted["arrival_rate"].to_numpy(dtype=float)
    if float(base_counts.sum()) <= 0:
        raise ValueError("Hourly arrival rates must contain at least one positive value.")

    raw = base_counts / base_counts.sum() * target_tasks
    counts = np.floor(raw).astype(int)
    remainder = int(target_tasks - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    targeted["arrival_rate"] = counts
    return targeted


def _estimate_bytes_per_task(hourly_df: pd.DataFrame, base_cfg: dict, experiment_cfg: dict, seed: int) -> float:
    """估算任务 CSV 中单条任务记录的平均字节数。"""
    configured_tasks_path = _project_path(experiment_cfg["paths"]["tasks_path"])
    candidate_paths = [
        configured_tasks_path,
        PROJECT_ROOT / "data" / "synthetic" / "tasks.csv",
    ]
    for path in candidate_paths:
        row_count = _count_csv_rows(path)
        if row_count > 0:
            return max(path.stat().st_size / row_count, 1.0)

    with tempfile.TemporaryDirectory(prefix="large_synthetic_sample_") as tmp_dir:
        sample_path = Path(tmp_dir) / "tasks_sample.csv"
        sample_df = generate_synthetic_tasks(
            hourly_df=_hourly_for_scale(hourly_df, 1.0),
            output_path=sample_path,
            seed=seed,
            max_delay_slots_tolerant=int(base_cfg["constraints"]["max_delay_slots_tolerant"]),
            max_delay_slots_token=int(base_cfg["constraints"]["max_delay_slots_token"]),
        )
        return max(sample_path.stat().st_size / max(len(sample_df), 1), 1.0)


def _write_report(tasks_df: pd.DataFrame, output_path: Path, generation_mode: str) -> None:
    """写入真实场景构建报告。"""
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    type_counts = tasks_df["task_type"].value_counts().reindex(
        ["delay_sensitive", "delay_tolerant", "token_batch"],
        fill_value=0,
    )
    gpu_mask = (tasks_df["gpu_demand"] > 0) | (tasks_df["task_type"] == "token_batch")
    hourly_counts = tasks_df.groupby("arrival_time").size().reindex(range(24), fill_value=0)

    lines = [
        "大规模合成任务数据报告",
        "",
        f"生成模式: {generation_mode}",
        f"输出文件: {output_path}",
        f"任务总数: {len(tasks_df)}",
        f"CSV 文件大小 MB: {output_size_mb:.4f}",
        "",
        "各类任务数量:",
        f"- delay_sensitive: {int(type_counts['delay_sensitive'])}",
        f"- delay_tolerant: {int(type_counts['delay_tolerant'])}",
        f"- token_batch: {int(type_counts['token_batch'])}",
        "",
        "CPU/GPU 任务数量:",
        f"- CPU 任务: {int((~gpu_mask).sum())}",
        f"- GPU 任务: {int(gpu_mask.sum())}",
        "",
        "各小时任务数量分布:",
        hourly_counts.to_string(),
        "",
        "平均资源需求:",
        f"- cpu_demand: {float(tasks_df['cpu_demand'].mean()):.6f}",
        f"- gpu_demand: {float(tasks_df['gpu_demand'].mean()):.6f}",
        f"- memory_demand: {float(tasks_df['memory_demand'].mean()):.6f}",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回脚本运行配置。"""
    parser = argparse.ArgumentParser(description="生成大规模标准合成任务 CSV，用于扩展实验和运行时间压力测试。")
    generation = parser.add_mutually_exclusive_group()
    generation.add_argument("--scale", type=float, default=1.0, help="负载放大倍数，例如 1、5、10、20。")
    generation.add_argument("--target-tasks", type=int, help="指定目标任务数量，例如 50000、100000。")
    generation.add_argument("--target-mb", type=float, help="指定目标 CSV 大小 MB，例如 64。")
    parser.add_argument("--output", default="data/synthetic/tasks_large.csv", help="输出任务 CSV 路径。")
    parser.add_argument("--seed", type=int, help="随机种子；默认读取 config/experiment.yaml。")
    return parser.parse_args()


def main() -> None:
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    args = parse_args()
    configs = load_all_configs(PROJECT_ROOT / "config")
    base_cfg = configs["base"]
    experiment_cfg = configs["experiment"]
    seed = int(args.seed if args.seed is not None else experiment_cfg["seed"])

    hourly_df = _load_hourly_profile(configs)
    if args.target_mb is not None:
        if args.target_mb <= 0:
            raise ValueError("--target-mb must be greater than 0.")
        bytes_per_task = _estimate_bytes_per_task(hourly_df, base_cfg, experiment_cfg, seed)
        target_tasks = max(1, int(round(args.target_mb * 1024 * 1024 / bytes_per_task)))
        generation_mode = f"target_mb={args.target_mb:g}, estimated_target_tasks={target_tasks}"
        generation_hourly_df = _hourly_for_target_tasks(hourly_df, target_tasks)
    elif args.target_tasks is not None:
        generation_mode = f"target_tasks={args.target_tasks}"
        generation_hourly_df = _hourly_for_target_tasks(hourly_df, int(args.target_tasks))
    else:
        generation_mode = f"scale={args.scale:g}"
        generation_hourly_df = _hourly_for_scale(hourly_df, float(args.scale))

    output_path = _project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df = generate_synthetic_tasks(
        hourly_df=generation_hourly_df,
        output_path=output_path,
        seed=seed,
        max_delay_slots_tolerant=int(base_cfg["constraints"]["max_delay_slots_tolerant"]),
        max_delay_slots_token=int(base_cfg["constraints"]["max_delay_slots_token"]),
    )
    _write_report(tasks_df, output_path, generation_mode)

    print(f"Tasks written to: {output_path}")
    print(f"Task count: {len(tasks_df)}")
    print(f"CSV size MB: {output_path.stat().st_size / (1024 * 1024):.4f}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
