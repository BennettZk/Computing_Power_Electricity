from __future__ import annotations

import numpy as np
import pandas as pd


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
