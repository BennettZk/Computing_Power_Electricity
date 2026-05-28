from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULT_CSV_COLUMN_MAPPING = {
    "algorithm": "算法",
    "scenario": "场景",
    "total_energy_kwh": "总能耗/kWh",
    "total_cost": "总成本",
    "total_carbon": "总碳排放/kg",
    "avg_delay_hours": "平均时延/h",
    "sla_violation_rate": "SLA违约率",
    "completion_rate": "任务完成率",
    "avg_cpu_utilization": "CPU平均利用率",
    "avg_gpu_utilization": "GPU平均利用率",
    "load_imbalance": "负载不均衡度",
    "peak_valley_gap_kw": "峰谷差/kW",
    "remote_task_count": "远端迁移任务数",
    "remote_completion_rate": "远端任务完成率",
    "remote_cost": "远端执行成本",
    "remote_energy_kwh": "远端网络能耗/kWh",
    "migration_delay_hours": "迁移平均时延/h",
    "unit_token_energy_kwh_per_million": "单位百万Token能耗/kWh",
    "unit_token_cost_per_million": "单位百万Token成本",
    "total_tokens": "Token总量",
    "delay_sensitive_violation_rate": "时延敏感任务违约率",
    "power_limit_violation_hours": "功率上限违约时段数",
    "peak_limit_violation_hours": "峰时功率违约时段数",
    "total_completed_tasks": "完成任务数",
    "total_tasks": "任务总数",
    "objective_penalty": "约束惩罚项",
    "hour": "小时",
    "price": "电价",
    "cpu_it_power_kw": "CPU IT功率/kW",
    "gpu_it_power_kw": "GPU IT功率/kW",
    "cooling_power_kw": "制冷功率/kW",
    "fixed_power_kw": "固定功率/kW",
    "total_power_kw": "本地总功率/kW",
    "effective_power_kw": "等效总功率/kW",
    "scale_or_tasks_path": "数据路径/规模",
    "task_count": "任务数量",
    "csv_size_mb": "CSV大小/MB",
    "mode": "模式",
    "step_name": "步骤名称",
    "elapsed_seconds": "耗时/s",
    "repeat_index": "重复编号",
    "n_jobs": "并行进程数",
    "cpu_count": "CPU核心数",
    "speedup_vs_single_process": "相对单进程加速比",
    "window_start": "窗口起始小时",
    "window_end": "窗口结束小时",
    "parameter": "参数",
    "value": "参数值",
    "scope": "范围",
    "optimized_task_count": "优化任务数",
    "solution_id": "解编号",
    "cpu_servers": "CPU开机数序列",
    "gpu_servers": "GPU开机数序列",
    "defer_ratio": "时间迁移比例序列",
    "migration_ratio": "空间迁移比例序列",
    "fitness": "适应度",
    "configured_migration_delay_hours": "配置跨区时延/h",
}


def export_csv_chinese(df: pd.DataFrame, path: str | Path, column_mapping: dict[str, str]) -> None:
    """
    复制 DataFrame，将英文列名按 column_mapping 映射为中文，然后导出 CSV。
    不修改原始 df。
    编码使用 utf-8-sig，方便 Excel 正常打开中文。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    export_df = export_df.rename(columns={col: column_mapping[col] for col in export_df.columns if col in column_mapping})
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_load_imbalance(cpu_utilization: list[float], gpu_utilization: list[float]) -> float:
    if not cpu_utilization or not gpu_utilization:
        return 0.0
    paired = zip(cpu_utilization, gpu_utilization)
    return float(np.mean([abs(cpu - gpu) for cpu, gpu in paired]))


def compute_peak_valley_gap(hourly_power_kw: list[float]) -> float:
    if not hourly_power_kw:
        return 0.0
    return float(max(hourly_power_kw) - min(hourly_power_kw))


def select_compromise_solution(df: pd.DataFrame, objective_cols: list[str]) -> pd.Series:
    norm = df[objective_cols].copy()
    for col in objective_cols:
        col_min = norm[col].min()
        col_max = norm[col].max()
        span = max(col_max - col_min, 1e-9)
        norm[col] = (norm[col] - col_min) / span
    score = norm.sum(axis=1)
    return df.loc[score.idxmin()]


def summarize_result_rows(result_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(result_rows)
    order = [
        "algorithm",
        "total_energy_kwh",
        "total_cost",
        "avg_delay_hours",
        "sla_violation_rate",
        "completion_rate",
        "avg_cpu_utilization",
        "avg_gpu_utilization",
        "load_imbalance",
        "peak_valley_gap_kw",
        "remote_task_count",
        "remote_completion_rate",
        "remote_cost",
        "remote_energy_kwh",
        "migration_delay_hours",
        "unit_token_energy_kwh_per_million",
        "unit_token_cost_per_million",
    ]
    remaining = [col for col in df.columns if col not in order]
    return df[order + remaining]
