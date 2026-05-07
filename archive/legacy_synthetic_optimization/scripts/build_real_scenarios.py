from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.task import load_tasks_csv


DEFAULT_CARBON = [
    0.52,
    0.51,
    0.50,
    0.49,
    0.49,
    0.50,
    0.53,
    0.56,
    0.61,
    0.64,
    0.66,
    0.67,
    0.65,
    0.63,
    0.62,
    0.63,
    0.66,
    0.69,
    0.71,
    0.70,
    0.64,
    0.60,
    0.56,
    0.54,
]
DEFAULT_RENEWABLE = [
    0.36,
    0.38,
    0.40,
    0.42,
    0.45,
    0.43,
    0.30,
    0.28,
    0.24,
    0.22,
    0.21,
    0.20,
    0.26,
    0.28,
    0.30,
    0.27,
    0.22,
    0.20,
    0.18,
    0.19,
    0.26,
    0.30,
    0.34,
    0.35,
]
DEFAULT_PRICE = [
    0.32,
    0.32,
    0.32,
    0.32,
    0.32,
    0.32,
    0.88,
    0.88,
    1.18,
    1.18,
    1.18,
    1.18,
    0.88,
    0.88,
    0.88,
    0.88,
    1.18,
    1.18,
    1.18,
    1.18,
    0.88,
    0.88,
    0.32,
    0.32,
]
REQUIRED_TASK_COLS = [
    "task_id",
    "arrival_time",
    "task_type",
    "cpu_demand",
    "gpu_demand",
    "memory_demand",
    "bandwidth_demand",
    "token_amount",
    "deadline",
    "priority",
    "migratable",
    "migration_cost_weight",
    "migration_delay_penalty",
]


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_input_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "real_case" / "server_tasks_24h.csv",
        PROJECT_ROOT / "data" / "real" / "tasks.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No real task CSV found. Expected data/real_case/server_tasks_24h.csv or data/real/tasks.csv. "
        "Run scripts/prepare_uploaded_case_dataset.py first, or pass --input."
    )


def _load_base_hourly() -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "real_case" / "hourly_input_24h.csv",
        PROJECT_ROOT / "data" / "real" / "hourly_input.csv",
        PROJECT_ROOT / "data" / "hourly_input_backup_before_real_case.csv",
        PROJECT_ROOT / "data" / "hourly_input.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path).sort_values("hour").reset_index(drop=True)
            if "hour" in df.columns and len(df) >= 24:
                return df.head(24)
    return pd.DataFrame({"hour": range(24), "price": DEFAULT_PRICE, "carbon_factor": DEFAULT_CARBON, "renewable_ratio": DEFAULT_RENEWABLE})


def _as_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series([default] * 0)
    if series.dtype == bool:
        return series.fillna(default)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _series_or_default(df: pd.DataFrame, col: str, default) -> pd.Series:
    if col in df.columns:
        return df[col]
    if isinstance(default, pd.Series):
        return default.reset_index(drop=True)
    return pd.Series([default] * len(df))


def _scale_cpu_demand(raw_cpu: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw_cpu, errors="coerce").fillna(1.0).clip(lower=0.0)
    q95 = max(float(numeric.quantile(0.95)), 1.0)
    scaled = 0.45 + np.clip(numeric / q95, 0.0, 1.0) * 2.05
    return pd.Series(scaled).round(4)


def _enhance_tasks(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Input task CSV is empty.")

    result = pd.DataFrame()
    result["task_id"] = df["task_id"].astype(str) if "task_id" in df.columns else [f"real-task-{idx:06d}" for idx in range(n_rows)]
    arrival = pd.to_numeric(df.get("arrival_time", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
    result["arrival_time"] = arrival

    raw_cpu = pd.to_numeric(_series_or_default(df, "cpu_demand", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    result["cpu_demand"] = _scale_cpu_demand(raw_cpu)
    raw_memory = pd.to_numeric(_series_or_default(df, "memory_demand", raw_cpu * 0.5), errors="coerce").fillna(raw_cpu * 0.5).clip(lower=0.0)
    memory_q95 = max(float(raw_memory.quantile(0.95)), 1.0)
    result["memory_demand"] = (4.0 + np.clip(raw_memory / memory_q95, 0.0, 1.0) * 14.0).round(4)
    result["bandwidth_demand"] = pd.to_numeric(_series_or_default(df, "bandwidth_demand", 0.5), errors="coerce").fillna(0.5).clip(lower=0.1, upper=2.0)

    heavy_score = raw_cpu.rank(pct=True) * 0.75 + raw_memory.rank(pct=True) * 0.25
    legal_types = {"delay_sensitive", "delay_tolerant", "token_batch"}
    if "task_type" in df.columns:
        task_type = df["task_type"].astype(str).where(df["task_type"].astype(str).isin(legal_types), "delay_sensitive")
        # Uploaded real case has no GPU/token semantics. Promote the heaviest tolerant jobs to token_batch
        # so the real scaled scenarios still exercise heterogeneous GPU/token behavior.
        if (task_type == "token_batch").sum() == 0:
            promote = (task_type != "delay_sensitive") & (heavy_score >= heavy_score.quantile(0.80))
            task_type = task_type.mask(promote, "token_batch")
    else:
        task_type = pd.Series("delay_sensitive", index=df.index)
        task_type = task_type.mask(heavy_score >= heavy_score.quantile(0.80), "token_batch")
        task_type = task_type.mask((heavy_score >= heavy_score.quantile(0.50)) & (heavy_score < heavy_score.quantile(0.80)), "delay_tolerant")
    result["task_type"] = task_type.to_numpy()

    gpu = pd.to_numeric(_series_or_default(df, "gpu_demand", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    token_mask = result["task_type"] == "token_batch"
    tolerant_mask = result["task_type"] == "delay_tolerant"
    sensitive_mask = result["task_type"] == "delay_sensitive"
    token_gpu_selected = token_mask & (rng.random(n_rows) < 0.75)
    tolerant_gpu_selected = tolerant_mask & (rng.random(n_rows) < 0.08)
    gpu = gpu.mask(token_gpu_selected, rng.uniform(0.4, 1.5, n_rows))
    gpu = gpu.mask(tolerant_gpu_selected, rng.uniform(0.2, 0.6, n_rows))
    gpu = gpu.mask(sensitive_mask, 0.0)
    result["gpu_demand"] = gpu.round(4)

    token_amount = pd.to_numeric(_series_or_default(df, "token_amount", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    token_base = 500.0 + np.clip(heavy_score.to_numpy(), 0.0, 1.0) * 4500.0
    token_noise = rng.uniform(0.85, 1.15, n_rows)
    token_amount = token_amount.mask(token_mask & (token_amount <= 0.0), token_base * token_noise)
    token_amount = token_amount.mask(~token_mask, 0.0)
    result["token_amount"] = token_amount.round(3)

    if "migratable" in df.columns:
        existing_migratable = _as_bool_series(df["migratable"]).reset_index(drop=True)
    else:
        existing_migratable = pd.Series([False] * n_rows)
    migratable = existing_migratable.copy()
    migratable = migratable.mask(sensitive_mask, False)
    migratable = migratable | (tolerant_mask & (rng.random(n_rows) < 0.25))
    migratable = migratable | (token_mask & (rng.random(n_rows) < 0.42))
    result["migratable"] = migratable.astype(bool)
    result["migration_cost_weight"] = pd.to_numeric(_series_or_default(df, "migration_cost_weight", 1.0), errors="coerce").fillna(1.0).clip(lower=0.1, upper=3.0)
    delay_penalty = pd.to_numeric(_series_or_default(df, "migration_delay_penalty", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    generated_penalty = np.where(token_mask, rng.uniform(0.08, 0.20, n_rows), rng.uniform(0.02, 0.10, n_rows))
    delay_penalty = delay_penalty.mask(result["migratable"] & (delay_penalty <= 0.0), generated_penalty)
    delay_penalty = delay_penalty.mask(~result["migratable"], 0.0)
    result["migration_delay_penalty"] = delay_penalty.round(4)

    original_deadline = pd.to_numeric(_series_or_default(df, "deadline", arrival + 1), errors="coerce").fillna(arrival + 1).astype(int)
    deadline = original_deadline.copy()
    deadline = deadline.mask(sensitive_mask, np.minimum(arrival + 1, 23))
    deadline = deadline.mask(tolerant_mask, np.maximum(deadline, arrival + 3))
    deadline = deadline.mask(token_mask, np.maximum(deadline, arrival + 4))
    result["deadline"] = np.maximum(deadline, arrival).clip(0, 23).astype(int)
    result["priority"] = np.select([sensitive_mask, tolerant_mask, token_mask], [3, 2, 1], default=2).astype(int)

    return result[REQUIRED_TASK_COLS]


def _target_count_by_hour(hour_counts: pd.Series, target_size: int) -> pd.Series:
    total = int(hour_counts.sum())
    if target_size >= total:
        return hour_counts.astype(int)
    raw = hour_counts / max(total, 1) * target_size
    counts = np.floor(raw).astype(int)
    remainder = int(target_size - counts.sum())
    if remainder > 0:
        for hour in (raw - counts).sort_values(ascending=False).index[:remainder]:
            counts.loc[hour] += 1
    return pd.Series(np.minimum(counts, hour_counts), index=hour_counts.index).astype(int)


def _sample_by_hour(tasks: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    hour_counts = tasks.groupby("arrival_time").size().reindex(range(24), fill_value=0)
    target_counts = _target_count_by_hour(hour_counts, target_size)
    sampled_parts: list[pd.DataFrame] = []
    for hour, count in target_counts.items():
        hour_df = tasks[tasks["arrival_time"] == hour]
        if count <= 0 or hour_df.empty:
            continue
        sampled_parts.append(hour_df.sample(n=int(count), random_state=seed + int(hour), replace=False))
    if not sampled_parts:
        return tasks.head(0).copy()
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)


def _hourly_for_tasks(tasks: pd.DataFrame, base_hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = pd.DataFrame({"hour": range(24)})
    hourly["arrival_rate"] = tasks.groupby("arrival_time").size().reindex(range(24), fill_value=0).to_numpy(dtype=int)
    base = base_hourly.set_index("hour").reindex(range(24))
    hourly["price"] = pd.to_numeric(base.get("price", pd.Series(DEFAULT_PRICE, index=range(24))), errors="coerce").fillna(pd.Series(DEFAULT_PRICE, index=range(24))).to_numpy(dtype=float)
    hourly["carbon_factor"] = pd.to_numeric(base.get("carbon_factor", pd.Series(DEFAULT_CARBON, index=range(24))), errors="coerce").fillna(pd.Series(DEFAULT_CARBON, index=range(24))).to_numpy(dtype=float)
    hourly["renewable_ratio"] = pd.to_numeric(base.get("renewable_ratio", pd.Series(DEFAULT_RENEWABLE, index=range(24))), errors="coerce").fillna(pd.Series(DEFAULT_RENEWABLE, index=range(24))).to_numpy(dtype=float)
    return hourly


def _scenario_name(target: str) -> str:
    return "full" if target.lower() == "full" else f"{int(target) // 1000}k"


def _write_report(rows: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于真实案例任务构建 real_5k / real_10k / real_full 缩放场景。")
    parser.add_argument("--input", help="标准任务 CSV。默认优先 data/real_case/server_tasks_24h.csv，再尝试 data/real/tasks.csv。")
    parser.add_argument("--output-dir", default="data/real_scaled", help="输出目录。")
    parser.add_argument("--target-sizes", nargs="+", default=["5000", "10000", "full"], help="目标规模，例如 5000 10000 full。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = _project_path(args.input) if args.input else _default_input_path()
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_tasks = pd.read_csv(input_path)
    enhanced = _enhance_tasks(raw_tasks, args.seed)
    base_hourly = _load_base_hourly()

    report_lines = [
        "真实集群级数据缩放场景报告",
        "",
        f"输入文件: {input_path}",
        f"原始任务数: {len(raw_tasks)}",
        f"增强后 token_batch 数量: {int((enhanced['task_type'] == 'token_batch').sum())}",
        f"增强后 GPU 任务数量: {int((enhanced['gpu_demand'] > 0).sum())}",
        f"增强后可迁移任务数量: {int(enhanced['migratable'].sum())}",
        f"增强后 Token 总量: {float(enhanced['token_amount'].sum()):.3f}",
        "",
    ]

    for target in args.target_sizes:
        scenario = _scenario_name(target)
        scenario_tasks = enhanced.copy() if target.lower() == "full" else _sample_by_hour(enhanced, int(target), args.seed)
        scenario_hourly = _hourly_for_tasks(scenario_tasks, base_hourly)
        tasks_path = output_dir / f"tasks_{scenario}.csv"
        hourly_path = output_dir / f"hourly_input_{scenario}.csv"
        scenario_tasks.to_csv(tasks_path, index=False, encoding="utf-8-sig")
        scenario_hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
        load_tasks_csv(tasks_path)

        report_lines.extend(
            [
                f"场景: real_{scenario}",
                f"- 任务文件: {tasks_path}",
                f"- 小时输入: {hourly_path}",
                f"- 任务数: {len(scenario_tasks)}",
                f"- 文件大小MB: {tasks_path.stat().st_size / (1024 * 1024):.4f}",
                f"- token_batch: {int((scenario_tasks['task_type'] == 'token_batch').sum())}",
                f"- GPU任务数: {int((scenario_tasks['gpu_demand'] > 0).sum())}",
                f"- 可迁移任务数: {int(scenario_tasks['migratable'].sum())}",
                f"- Token总量: {float(scenario_tasks['token_amount'].sum()):.3f}",
                "- 24小时任务分布:",
                scenario_hourly[["hour", "arrival_rate"]].to_string(index=False),
                "",
            ]
        )

    _write_report(report_lines, PROJECT_ROOT / "outputs" / "real_scaled_report.txt")
    print(f"Real scaled scenarios written to: {output_dir}")
    print("Report written to: outputs/real_scaled_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
