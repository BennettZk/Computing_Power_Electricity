from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STRATEGY_VALUE_MAPPING = {
    "No-Export": "不出口",
    "Power-Margin-Only": "仅功率裕度",
    "Price-Driven": "电价驱动",
    "Latency-Aware": "时延感知",
    "RH-TEO": "RH-TEO策略",
}

REGION_VALUE_MAPPING = {
    "Asia": "亚洲",
    "Europe": "欧洲",
    "America": "美洲",
}

PARAMETER_VALUE_MAPPING = {
    "base_token_sla_hours": "SLA阈值",
    "latency_scale": "跨时区时延放大系数",
    "america_price_multiplier": "美洲价格倍率",
    "window_size_hours": "滚动窗口长度",
    "token_per_margin_unit_factor": "Token产能系数",
}

RESULT_CSV_COLUMN_MAPPING = {
    "algorithm": "算法",
    "strategy": "策略",
    "scenario": "场景",
    "hour": "小时",
    "price": "电价",
    "power_cap_pu": "功率上限标幺值",
    "hourly_power_cap": "小时功率上限",
    "high_price_flag": "高电价标记",
    "low_price_flag": "低电价标记",
    "hourly_task_arrivals": "小时任务到达量",
    "hourly_cpu_demand": "小时CPU需求",
    "hourly_completed_tasks": "小时完成任务数",
    "hourly_deadline_violation_rate": "小时Deadline违约率",
    "arrival_rate": "任务到达率",
    "hourly_actual_power_w": "小时实际功率/W",
    "hourly_reported_power_w": "小时报量功率/W",
    "hourly_power_cap_w": "小时芯片功率上限/W",
    "hourly_frequency_ghz": "小时频率/GHz",
    "dvfs_tracking_error": "DVFS跟踪误差",
    "cluster_cap_norm": "集群功率上限归一化",
    "server_load_norm": "服务器负载归一化",
    "chip_power_norm": "芯片功率最大值归一化",
    "chip_power_variation_norm": "芯片功率波动归一化",
    "chip_power_ratio": "芯片功率占上限比例",
    "power_margin_norm_old": "旧等效功率裕度",
    "power_margin_norm_v2": "新等效功率裕度",
    "power_margin_norm": "等效功率裕度",
    "alpha": "服务器负载权重",
    "beta": "芯片功率波动权重",
    "base_reserve": "基础预留裕度",
    "token_capacity": "Token产出能力",
    "export_tokens": "出口Token量",
    "total_export_tokens": "总出口Token量",
    "token_export_revenue": "Token出口收益",
    "token_energy_cost": "Token电力成本",
    "net_token_profit": "Token净收益",
    "avg_cross_timezone_delay": "平均跨时区时延/h",
    "token_sla_violation_rate": "Token SLA违约率",
    "energy_per_million_tokens": "百万Token能耗/kWh",
    "cost_per_million_tokens": "百万Token成本",
    "asia_export_tokens": "亚洲出口Token量",
    "europe_export_tokens": "欧洲出口Token量",
    "america_export_tokens": "美洲出口Token量",
    "america_export_share": "美洲出口占比",
    "region": "地区",
    "share": "占比",
    "parameter": "参数",
    "value": "参数值",
    "metric": "指标",
    "utility": "综合效用",
    "export_ramp_tokens": "出口波动Token量",
    "server_room_id": "机房编号",
    "rack_id": "机架编号",
    "server_id": "服务器编号",
    "task_count": "任务数量",
    "cpu_usage": "CPU使用量",
    "remark": "备注",
}


def localize_strategy_values(df: pd.DataFrame, column: str = "strategy") -> pd.DataFrame:
    output = df.copy()
    if column in output.columns:
        output[column] = output[column].map(STRATEGY_VALUE_MAPPING).fillna(output[column])
    return output


def localize_region_values(df: pd.DataFrame, column: str = "region") -> pd.DataFrame:
    output = df.copy()
    if column in output.columns:
        output[column] = output[column].map(REGION_VALUE_MAPPING).fillna(output[column])
    return output


def localize_parameter_values(df: pd.DataFrame, column: str = "parameter") -> pd.DataFrame:
    output = df.copy()
    if column in output.columns:
        output[column] = output[column].map(PARAMETER_VALUE_MAPPING).fillna(output[column])
    return output


def add_strategy_remark(df: pd.DataFrame, column: str = "strategy") -> pd.DataFrame:
    output = df.copy()
    if column in output.columns:
        output["remark"] = np.where(
            output[column].isin(["No-Export", "不出口"]),
            "零出口基准，仅用于表格对照，不参与主要图表展示",
            "实际Token出口策略",
        )
    return output


def export_csv_chinese(df: pd.DataFrame, path: str | Path, column_mapping: dict[str, str] | None = None) -> None:
    """
    Copy a DataFrame, rename known English columns to Chinese, and export it.

    The original df is not modified. utf-8-sig keeps Chinese headers readable in
    Excel on Windows.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = column_mapping or RESULT_CSV_COLUMN_MAPPING
    export_df = df.copy()
    export_df = export_df.rename(columns={col: mapping[col] for col in export_df.columns if col in mapping})
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def normalize_by_max(values: pd.Series | np.ndarray) -> pd.Series:
    """Normalize a non-negative series by its maximum; constants remain meaningful."""
    series = pd.Series(values, dtype="float64").fillna(0.0)
    max_value = float(series.max())
    if max_value <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series / max_value).clip(lower=0.0, upper=1.0)


def normalize_minmax(values: pd.Series | np.ndarray) -> pd.Series:
    """Min-max normalize a series; constants are mapped to all zeros."""
    series = pd.Series(values, dtype="float64").fillna(0.0)
    min_value = float(series.min())
    max_value = float(series.max())
    span = max_value - min_value
    if span <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - min_value) / span).clip(lower=0.0, upper=1.0)


def minmax_norm(values: pd.Series | np.ndarray) -> pd.Series:
    return normalize_minmax(values)
