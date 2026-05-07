from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRICE = [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.88, 0.88, 1.18, 1.18, 1.18, 1.18, 0.88, 0.88, 0.88, 0.88, 1.18, 1.18, 1.18, 1.18, 0.88, 0.88, 0.32, 0.32]
DEFAULT_CARBON = [0.52, 0.51, 0.50, 0.49, 0.49, 0.50, 0.53, 0.56, 0.61, 0.64, 0.66, 0.67, 0.65, 0.63, 0.62, 0.63, 0.66, 0.69, 0.71, 0.70, 0.64, 0.60, 0.56, 0.54]
DEFAULT_RENEWABLE = [0.36, 0.38, 0.40, 0.42, 0.45, 0.43, 0.30, 0.28, 0.24, 0.22, 0.21, 0.20, 0.26, 0.28, 0.30, 0.27, 0.22, 0.20, 0.18, 0.19, 0.26, 0.30, 0.34, 0.35]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _has_col(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in df.columns


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"WARNING: {message}")


def _guess_columns(columns: list[str], keywords: list[str]) -> list[str]:
    lowered = {col: col.lower() for col in columns}
    return [col for col, lower in lowered.items() if any(keyword in lower for keyword in keywords)]


def build_profile(df: pd.DataFrame, input_path: Path) -> str:
    columns = list(df.columns)
    lines: list[str] = [
        "真实数据字段画像",
        "",
        f"数据文件路径: {input_path}",
        f"行数: {len(df)}",
        f"列数: {len(columns)}",
        "",
        "所有列名:",
        ", ".join(columns) if columns else "(无列)",
        "",
        "列详情:",
    ]
    for col in columns:
        non_null_samples = df[col].dropna().astype(str).head(5).tolist()
        lines.extend(
            [
                f"- {col}",
                f"  dtype: {df[col].dtype}",
                f"  缺失率: {float(df[col].isna().mean()):.4f}",
                f"  前5个非空样例: {non_null_samples}",
            ]
        )

    guess_groups = [
        ("可能的时间列", ["time", "timestamp", "submit", "arrival", "date", "hour"]),
        ("可能的任务ID列", ["task", "job", "id"]),
        ("可能的 CPU 字段", ["cpu", "core", "processor"]),
        ("可能的 GPU 字段", ["gpu", "accelerator", "cuda"]),
        ("可能的内存字段", ["mem", "memory", "ram"]),
        ("可能的功耗/能耗字段", ["power", "energy", "watt", "kwh"]),
        ("可能的任务持续时间或截止时间字段", ["duration", "runtime", "deadline", "finish", "end", "slack"]),
    ]
    lines.extend(["", "自动字段猜测:"])
    for title, keywords in guess_groups:
        guesses = _guess_columns(columns, keywords)
        lines.append(f"- {title}: {guesses if guesses else '未发现'}")
    return "\n".join(lines)


def _numeric_series(df: pd.DataFrame, col: str | None, default: float, warnings: list[str], fill_counter: dict[str, int], name: str) -> pd.Series:
    if not _has_col(df, col):
        _warn(warnings, f"{name} 字段缺失，使用默认值 {default}。")
        fill_counter[name] = len(df)
        return pd.Series([default] * len(df), index=df.index, dtype=float)

    raw = pd.to_numeric(df[col], errors="coerce")
    missing = int(raw.isna().sum())
    if missing:
        _warn(warnings, f"{name} 字段 {col} 有 {missing} 个缺失/非法值，使用默认值 {default} 填补。")
        fill_counter[name] = missing
    return raw.fillna(default).astype(float).clip(lower=0.0)


def _text_series(df: pd.DataFrame, col: str | None, default_prefix: str, warnings: list[str], fill_counter: dict[str, int], name: str) -> pd.Series:
    if not _has_col(df, col):
        _warn(warnings, f"{name} 字段缺失，使用 {default_prefix}-行号 自动生成。")
        fill_counter[name] = len(df)
        return pd.Series([f"{default_prefix}-{idx:06d}" for idx in range(len(df))], index=df.index)

    values = df[col].astype("string")
    missing = int(values.isna().sum())
    if missing:
        _warn(warnings, f"{name} 字段 {col} 有 {missing} 个缺失值，使用 {default_prefix}-行号 填补。")
        fill_counter[name] = missing
    fallback = pd.Series([f"{default_prefix}-{idx:06d}" for idx in range(len(df))], index=df.index, dtype="string")
    return values.fillna(fallback).astype(str)


def _map_time_to_hour(df: pd.DataFrame, time_col: str | None, warnings: list[str], fill_counter: dict[str, int]) -> pd.Series:
    n_rows = len(df)
    if n_rows == 0:
        return pd.Series([], dtype=int)
    if not _has_col(df, time_col):
        _warn(warnings, "时间列缺失，按行号均匀分配到 0-23 小时。")
        fill_counter["arrival_time"] = n_rows
        return pd.Series(np.floor(np.arange(n_rows) * 24 / n_rows).astype(int).clip(0, 23), index=df.index)

    values = df[time_col]
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_valid_ratio = 1.0 - float(numeric.isna().mean())
    if numeric_valid_ratio >= 0.8:
        filled = numeric.fillna(numeric.median() if not numeric.dropna().empty else 0)
        if filled.min() >= 0 and filled.max() <= 23:
            hours = np.rint(filled).astype(int).clip(0, 23)
        else:
            hours = np.mod(np.floor(filled).astype(int), 24)
        missing = int(numeric.isna().sum())
        if missing:
            _warn(warnings, f"时间列 {time_col} 有 {missing} 个非法数值，使用中位数填补后映射小时。")
            fill_counter["arrival_time"] = missing
        return pd.Series(hours, index=df.index, dtype=int)

    parsed = pd.to_datetime(values, errors="coerce")
    valid_ratio = 1.0 - float(parsed.isna().mean())
    if valid_ratio >= 0.5:
        missing = int(parsed.isna().sum())
        if missing:
            _warn(warnings, f"时间列 {time_col} 有 {missing} 个无法解析的 datetime，按行号补小时。")
            fill_counter["arrival_time"] = missing
        fallback_hours = pd.Series(np.floor(np.arange(n_rows) * 24 / n_rows).astype(int).clip(0, 23), index=df.index)
        parsed_hours = parsed.dt.hour
        return parsed_hours.fillna(fallback_hours).astype(int).clip(0, 23)

    _warn(warnings, f"时间列 {time_col} 无法可靠解析，按行号均匀分配到 0-23 小时。")
    fill_counter["arrival_time"] = n_rows
    return pd.Series(np.floor(np.arange(n_rows) * 24 / n_rows).astype(int).clip(0, 23), index=df.index)


def _infer_task_type(gpu: pd.Series, cpu: pd.Series, memory: pd.Series, duration: pd.Series, explicit: pd.Series | None = None) -> pd.Series:
    allowed = {"delay_sensitive", "delay_tolerant", "token_batch"}
    inferred = pd.Series(["delay_sensitive"] * len(gpu), index=gpu.index, dtype="object")
    inferred[gpu > 0] = "token_batch"

    duration_threshold = max(float(duration.quantile(0.75)), 2.0) if len(duration) else 2.0
    cpu_threshold = max(float(cpu.quantile(0.75)), 1.0) if len(cpu) else 1.0
    mem_threshold = max(float(memory.quantile(0.75)), 4.0) if len(memory) else 4.0
    heavy = (duration >= duration_threshold) | (cpu >= cpu_threshold) | (memory >= mem_threshold)
    inferred[(gpu <= 0) & heavy] = "delay_tolerant"

    if explicit is not None:
        normalized = explicit.astype("string").str.strip().str.lower()
        mapping = {
            "sensitive": "delay_sensitive",
            "delay_sensitive": "delay_sensitive",
            "latency": "delay_sensitive",
            "tolerant": "delay_tolerant",
            "delay_tolerant": "delay_tolerant",
            "batch": "token_batch",
            "token": "token_batch",
            "token_batch": "token_batch",
            "gpu": "token_batch",
        }
        mapped = normalized.map(mapping)
        inferred[mapped.isin(allowed)] = mapped[mapped.isin(allowed)]
    return inferred


def _priority_from_task_type(task_type: pd.Series, explicit_priority: pd.Series | None, warnings: list[str], fill_counter: dict[str, int]) -> pd.Series:
    default_priority = task_type.map({"delay_sensitive": 3, "delay_tolerant": 2, "token_batch": 1}).fillna(2).astype(int)
    if explicit_priority is None:
        fill_counter["priority"] = len(task_type)
        return default_priority
    numeric = pd.to_numeric(explicit_priority, errors="coerce")
    missing = int(numeric.isna().sum())
    if missing:
        _warn(warnings, f"priority 字段有 {missing} 个缺失/非法值，按 task_type 推断。")
        fill_counter["priority"] = missing
    return numeric.fillna(default_priority).round().astype(int).clip(1, 3)


def _build_deadline(
    arrival: pd.Series,
    duration: pd.Series,
    deadline_raw: pd.Series | None,
    warnings: list[str],
    fill_counter: dict[str, int],
) -> tuple[pd.Series, int]:
    if deadline_raw is None:
        deadline = arrival + np.maximum(np.ceil(duration).astype(int), 2)
        fill_counter["deadline"] = len(arrival)
    else:
        numeric = pd.to_numeric(deadline_raw, errors="coerce")
        missing = int(numeric.isna().sum())
        if missing:
            _warn(warnings, f"deadline 字段有 {missing} 个缺失/非法值，使用 arrival_time + duration/default 填补。")
            fill_counter["deadline"] = missing
        fallback = arrival + np.maximum(np.ceil(duration).astype(int), 2)
        deadline = numeric.fillna(fallback)
        if float(deadline.max()) > 23 or float(deadline.min()) < 0:
            deadline = np.where(deadline < 24, deadline, arrival + np.maximum(np.ceil(duration).astype(int), 2))

    deadline = pd.Series(np.maximum(np.rint(deadline).astype(int), arrival.astype(int)), index=arrival.index)
    truncated_count = int((deadline > 23).sum())
    deadline = deadline.clip(0, 23).astype(int)
    deadline = pd.Series(np.maximum(deadline, arrival.astype(int)), index=arrival.index).clip(0, 23).astype(int)
    return deadline, truncated_count


def _bool_series(df: pd.DataFrame, col: str | None, default: bool) -> pd.Series:
    if not _has_col(df, col):
        return pd.Series([default] * len(df), index=df.index)
    values = df[col]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def convert_dataset(args: argparse.Namespace, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    warnings: list[str] = []
    fill_counter: dict[str, int] = {}
    original_rows = len(df)

    arrival = _map_time_to_hour(df, args.time_col, warnings, fill_counter)
    task_id = _text_series(df, args.task_id_col, "real-task", warnings, fill_counter, "task_id")
    cpu = _numeric_series(df, args.cpu_col, 1.0, warnings, fill_counter, "cpu_demand")
    gpu = _numeric_series(df, args.gpu_col, 0.0, warnings, fill_counter, "gpu_demand")
    memory = _numeric_series(df, args.memory_col, 4.0, warnings, fill_counter, "memory_demand")
    bandwidth = _numeric_series(df, args.bandwidth_col, 0.5, warnings, fill_counter, "bandwidth_demand")
    token_amount = _numeric_series(df, args.token_col, 0.0, warnings, fill_counter, "token_amount")
    duration = _numeric_series(df, args.duration_col, 2.0, warnings, fill_counter, "duration")

    explicit_type = df[args.task_type_col] if _has_col(df, args.task_type_col) else None
    if explicit_type is None:
        fill_counter["task_type"] = original_rows
    task_type = _infer_task_type(gpu=gpu, cpu=cpu, memory=memory, duration=duration, explicit=explicit_type)
    explicit_priority = df[args.priority_col] if _has_col(df, args.priority_col) else None
    priority = _priority_from_task_type(task_type, explicit_priority, warnings, fill_counter)
    deadline_raw = df[args.deadline_col] if _has_col(df, args.deadline_col) else None
    deadline, truncated_count = _build_deadline(arrival, duration, deadline_raw, warnings, fill_counter)

    migratable = _bool_series(df, args.migratable_col, default=False)
    migration_cost_weight = _numeric_series(df, args.migration_cost_weight_col, 1.0, warnings, fill_counter, "migration_cost_weight")
    migration_delay_penalty = _numeric_series(df, args.migration_delay_penalty_col, 0.0, warnings, fill_counter, "migration_delay_penalty")

    tasks_df = pd.DataFrame(
        {
            "task_id": task_id,
            "arrival_time": arrival.astype(int),
            "task_type": task_type,
            "cpu_demand": cpu,
            "gpu_demand": gpu,
            "memory_demand": memory,
            "bandwidth_demand": bandwidth,
            "token_amount": token_amount,
            "deadline": deadline,
            "priority": priority,
            "migratable": migratable,
            "migration_cost_weight": migration_cost_weight,
            "migration_delay_penalty": migration_delay_penalty,
        }
    )

    hourly = tasks_df.groupby("arrival_time").size().reindex(range(24), fill_value=0).rename("arrival_rate").reset_index()
    hourly = hourly.rename(columns={"arrival_time": "hour"})
    hourly["price"] = _hourly_from_optional_col(df, args.price_col, arrival, DEFAULT_PRICE, warnings, "price")
    hourly["carbon_factor"] = _hourly_from_optional_col(df, args.carbon_col, arrival, DEFAULT_CARBON, warnings, "carbon_factor")
    hourly["renewable_ratio"] = _hourly_from_optional_col(df, args.renewable_col, arrival, DEFAULT_RENEWABLE, warnings, "renewable_ratio")

    report = build_cleaning_report(
        input_path=Path(args.input),
        original_rows=original_rows,
        tasks_df=tasks_df,
        hourly_df=hourly,
        output_tasks=Path(args.output_tasks),
        output_hourly=Path(args.output_hourly),
        warnings=warnings,
        fill_counter=fill_counter,
        deadline_truncated_count=truncated_count,
        profile_only=False,
    )
    return tasks_df, hourly, report


def _hourly_from_optional_col(
    df: pd.DataFrame,
    col: str | None,
    arrival: pd.Series,
    default_values: list[float],
    warnings: list[str],
    name: str,
) -> list[float]:
    if not _has_col(df, col):
        _warn(warnings, f"{name} 字段缺失，使用项目默认 24 小时曲线。")
        return default_values
    values = pd.to_numeric(df[col], errors="coerce")
    grouped = values.groupby(arrival).mean().reindex(range(24))
    default = pd.Series(default_values, index=range(24), dtype=float)
    missing = int(grouped.isna().sum())
    if missing:
        _warn(warnings, f"{name} 字段有 {missing} 个小时缺失，使用项目默认曲线补齐。")
    return grouped.fillna(default).astype(float).tolist()


def build_cleaning_report(
    input_path: Path,
    original_rows: int,
    tasks_df: pd.DataFrame | None,
    hourly_df: pd.DataFrame | None,
    output_tasks: Path | None,
    output_hourly: Path | None,
    warnings: list[str],
    fill_counter: dict[str, int],
    deadline_truncated_count: int,
    profile_only: bool,
) -> str:
    lines = [
        "真实数据清洗报告",
        "",
        f"输入文件: {input_path}",
        f"原始数据行数: {original_rows}",
        f"模式: {'profile-only' if profile_only else '字段映射转换'}",
        "",
    ]
    if tasks_df is None or hourly_df is None:
        lines.extend(
            [
                "转换后任务数: 0",
                "说明: profile-only 模式未生成标准 tasks/hourly CSV。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"转换后任务数: {len(tasks_df)}",
                "",
                "各小时任务数分布:",
                hourly_df[["hour", "arrival_rate"]].to_string(index=False),
                "",
                "task_type 分布:",
                tasks_df["task_type"].value_counts().to_string(),
                "",
                "资源需求统计:",
            ]
        )
        for col in ["cpu_demand", "gpu_demand", "memory_demand"]:
            lines.append(
                f"- {col}: mean={tasks_df[col].mean():.4f}, min={tasks_df[col].min():.4f}, max={tasks_df[col].max():.4f}"
            )
        lines.extend(
            [
                "",
                f"deadline 截断数量: {deadline_truncated_count}",
                "",
                "缺失值/默认值填补数量:",
            ]
        )
        if fill_counter:
            lines.extend([f"- {key}: {value}" for key, value in sorted(fill_counter.items())])
        else:
            lines.append("- 无")
        lines.extend(
            [
                "",
                f"tasks 输出路径: {output_tasks}",
                f"hourly 输出路径: {output_hourly}",
            ]
        )

    lines.extend(["", "Warnings:"])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- 无")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将未知字段真实数据画像或转换为项目标准输入 CSV。")
    parser.add_argument("--input", required=True, help="真实数据 CSV 路径。")
    parser.add_argument("--profile-only", action="store_true", help="只输出字段画像，不生成标准输入。")
    parser.add_argument("--output-tasks", default="data/real/tasks.csv", help="标准任务 CSV 输出路径。")
    parser.add_argument("--output-hourly", default="data/real/hourly_input.csv", help="标准小时输入 CSV 输出路径。")
    parser.add_argument("--profile-output", default="outputs/real_data_profile.txt", help="字段画像报告输出路径。")
    parser.add_argument("--report-output", default="outputs/real_data_cleaning_report.txt", help="清洗报告输出路径。")

    parser.add_argument("--time-col")
    parser.add_argument("--task-id-col")
    parser.add_argument("--task-type-col")
    parser.add_argument("--cpu-col")
    parser.add_argument("--gpu-col")
    parser.add_argument("--memory-col")
    parser.add_argument("--bandwidth-col")
    parser.add_argument("--token-col")
    parser.add_argument("--duration-col")
    parser.add_argument("--deadline-col")
    parser.add_argument("--priority-col")
    parser.add_argument("--migratable-col")
    parser.add_argument("--migration-cost-weight-col")
    parser.add_argument("--migration-delay-penalty-col")
    parser.add_argument("--price-col")
    parser.add_argument("--carbon-col")
    parser.add_argument("--renewable-col")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_profile = Path(args.profile_output)
    output_report = Path(args.report_output)
    _ensure_parent(output_profile)
    _ensure_parent(output_report)

    df = _read_csv(input_path)
    profile = build_profile(df, input_path)
    output_profile.write_text(profile, encoding="utf-8")

    if args.profile_only:
        report = build_cleaning_report(
            input_path=input_path,
            original_rows=len(df),
            tasks_df=None,
            hourly_df=None,
            output_tasks=None,
            output_hourly=None,
            warnings=[],
            fill_counter={},
            deadline_truncated_count=0,
            profile_only=True,
        )
        output_report.write_text(report, encoding="utf-8")
        print(f"Profile written to: {output_profile}")
        print(f"Cleaning report written to: {output_report}")
        return

    tasks_df, hourly_df, report = convert_dataset(args, df)
    output_tasks = Path(args.output_tasks)
    output_hourly = Path(args.output_hourly)
    _ensure_parent(output_tasks)
    _ensure_parent(output_hourly)
    tasks_df.to_csv(output_tasks, index=False, encoding="utf-8-sig")
    hourly_df.to_csv(output_hourly, index=False, encoding="utf-8-sig")
    output_report.write_text(report, encoding="utf-8")

    print(f"Profile written to: {output_profile}")
    print(f"Tasks written to: {output_tasks}")
    print(f"Hourly profile written to: {output_hourly}")
    print(f"Cleaning report written to: {output_report}")


if __name__ == "__main__":
    main()
