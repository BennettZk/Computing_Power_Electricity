# -*- coding: utf-8 -*-
"""
生成“跨域算力服务时空分布演化过程”GIF/MP4/封面图，并导出 DC-区域分配矩阵。

依赖安装命令：
    pip install pandas matplotlib openpyxl imageio pillow tqdm

说明：
    1. 本脚本采用抽象算力网拓扑，不使用真实地图，不构造经纬度。
    2. 优先读取已有 DC-区域分配矩阵；若项目中不存在逐 DC、逐区域、逐时间步
       的真实分配矩阵，则基于等效功率裕度构造可视化用分配场景。
    3. 构造场景会导出 output/dc_region_allocation_matrix.csv，分配量字段单位为 M Token，便于答辩解释。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm


TITLE = "跨域算力服务时空分布演化过程"
TREND_TITLE = "区域算力服务分配趋势"

OUTPUT_GIF_NAME = "computing_network_allocation_animation.gif"
OUTPUT_MP4_NAME = "computing_network_allocation_animation.mp4"
OUTPUT_COVER_NAME = "computing_network_allocation_animation_cover.png"
OUTPUT_MATRIX_NAME = "dc_region_allocation_matrix.csv"

DC_IDS = [0, 1, 2, 3]
REGIONS = ["本地", "中距离", "远距离"]
REGION_FILE_COLUMNS = {
    "本地": "local_allocation",
    "中距离": "medium_allocation",
    "远距离": "remote_allocation",
}
REGION_LATENCY_HOURS = np.array([0.08, 0.25, 0.42])
REGION_SLA_RISK = np.array([0.02, 0.12, 0.28])
TOKEN_UNIT_SCALE = 1_000_000
TOKEN_UNIT_LABEL = "M Token"

FIELD_CANDIDATES = {
    "time_step": ["time_step", "hour", "time", "timestamp", "t", "step", "时间步", "小时"],
    "datacenter_id": ["datacenter_id", "dc_id", "dc", "data_center", "datacenter", "数据中心"],
    "region": ["region", "service_region", "区域", "服务区域"],
    "allocation": ["allocation", "service_amount", "token_export", "export_tokens", "服务量", "分配量"],
    "local": ["local_allocation", "local", "domestic_export_tokens", "domestic", "本地"],
    "medium": ["medium_allocation", "medium", "europe_export_tokens", "europe", "中距离"],
    "remote": [
        "remote_allocation",
        "remote",
        "north_america_export_tokens",
        "northamerica",
        "north_america",
        "远距离",
    ],
    "total_service": ["export_tokens", "total_export_tokens", "total_service", "service_amount"],
    "strategy": ["strategy", "策略"],
    "latency": ["avg_cross_timezone_delay", "latency", "delay", "平均时延", "时延"],
    "sla_violation": ["token_sla_violation_rate", "sla_violation", "sla_violation_rate", "违约率"],
    "price": ["price", "power_price", "electricity_price", "电价"],
    "power_margin": [
        "power_margin_norm",
        "equivalent_power_margin",
        "power_margin",
        "normalized_power_cap",
        "cluster_cap_norm",
        "等效功率裕度",
    ],
    "dc_available_margin": ["dc_available_margin", "available_margin", "power_margin_norm", "裕度"],
}


@dataclass(frozen=True)
class OutputPaths:
    gif: Path
    mp4: Path
    cover: Path
    matrix: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回脚本运行配置。"""
    parser = argparse.ArgumentParser(description="生成 DC-区域跨域算力服务分配动图。")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录。")
    parser.add_argument("--frame-step", type=int, default=1, help="抽帧间隔，默认 1。")
    parser.add_argument("--max-frames", type=int, default=0, help="最多帧数，0 表示不限制。")
    parser.add_argument("--fps", type=int, default=4, help="动画帧率，默认 4 FPS。")
    parser.add_argument("--dpi", type=int, default=140, help="导出分辨率，默认 140 DPI。")
    parser.add_argument("--strategy", default="RH-TEO", help="若读取 token export 结果，优先使用的策略。")
    parser.add_argument("--skip-mp4", action="store_true", help="只导出 GIF。")
    return parser.parse_args()


def normalize_name(value: object) -> str:
    """标准化字段名称，消除符号、空格和大小写差异。"""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"[\s_\-./\\()（）\[\]【】{}:：,，;；]+", "", text)


def find_column(
    columns: Iterable[object],
    role: str,
    *,
    required: bool = False,
    source: str = "",
) -> str | None:
    """根据候选语义关键词在表头中查找匹配字段。"""
    original_columns = [str(col) for col in columns if col is not None and str(col) != "nan"]
    normalized = {normalize_name(col): col for col in original_columns}

    for candidate in FIELD_CANDIDATES[role]:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    for candidate in FIELD_CANDIDATES[role]:
        key = normalize_name(candidate)
        if not key or len(key) <= 3:
            continue
        for norm_col, original_col in normalized.items():
            if key in norm_col or norm_col in key:
                return original_col

    if required:
        raise ValueError(f"[字段缺失] {source} 缺少字段语义：{role}。已有字段：{original_columns}")
    return None


def to_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    """将输入序列转换为数值序列，并用默认值处理异常项。"""
    result = pd.to_numeric(series.replace("", np.nan), errors="coerce")
    if fill_value is not None:
        result = result.fillna(fill_value)
    return result


def finite_number(value: object) -> float | None:
    """判断输入值是否为有限数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def format_m_token(value: object) -> str:
    """把 Token 数量格式化为百万 Token 展示口径。"""
    number = finite_number(value)
    if number is None:
        return "无数据"
    return f"{number / TOKEN_UNIT_SCALE:.3f} {TOKEN_UNIT_LABEL}"


def format_percent(value: object) -> str:
    """把比例值格式化为百分比文本。"""
    number = finite_number(value)
    if number is None:
        return "无数据"
    return f"{number * 100:.1f}%"


def format_float(value: object, digits: int = 1, suffix: str = "") -> str:
    """按指定精度格式化浮点数，缺失值使用占位符。"""
    number = finite_number(value)
    if number is None:
        return "无数据"
    return f"{number:.{digits}f}{suffix}"


def configure_chinese_font() -> None:
    """配置 Matplotlib 中文字体，降低中文乱码风险。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用中文字体：{font_name}")
            return
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
    print("[字体提示] 未检测到 Microsoft YaHei/SimHei，已降级为默认字体。")


def discover_candidate_files(root: Path, output_matrix: Path) -> list[Path]:
    """在候选目录中查找可用的数据输入文件。"""
    keywords = [
        "regional_allocation",
        "rolling_horizon_results",
        "service_allocation",
        "dc_region_allocation",
        "allocation",
        "token_export_results",
        "hourly_token_export",
        "strategy_results",
    ]
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".json"}:
            continue
        if ".venv" in path.parts:
            continue
        if path.resolve() == output_matrix.resolve():
            continue
        lower_name = path.name.lower()
        if any(keyword in lower_name for keyword in keywords):
            candidates.append(path)
    return sorted(candidates, key=lambda p: (0 if "allocation" in p.name.lower() else 1, str(p)))


def map_region(value: object) -> str | None:
    """把区域文本映射为可视化使用的标准区域类别。"""
    text = normalize_name(value)
    if text in {"local", "domestic", "本地"}:
        return "本地"
    if text in {"medium", "middle", "europe", "中距离", "中程"}:
        return "中距离"
    if text in {"remote", "far", "northamerica", "northamerican", "远距离", "远程"}:
        return "远距离"
    return None


def try_load_real_matrix_from_csv(path: Path, strategy: str) -> pd.DataFrame | None:
    """尝试从 CSV 中读取真实或已生成的地区分配矩阵。"""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[跳过] {path} 读取失败：{exc}")
        return None

    print(f"[检查] {path} shape={df.shape}, columns={list(df.columns)}")
    if df.empty:
        return None

    strategy_col = find_column(df.columns, "strategy", source=str(path))
    if strategy_col is not None and strategy in set(df[strategy_col].astype(str)):
        df = df[df[strategy_col].astype(str) == strategy].copy()

    time_col = find_column(df.columns, "time_step", source=str(path))
    dc_col = find_column(df.columns, "datacenter_id", source=str(path))
    if time_col is None or dc_col is None:
        return None

    local_col = find_column(df.columns, "local", source=str(path))
    medium_col = find_column(df.columns, "medium", source=str(path))
    remote_col = find_column(df.columns, "remote", source=str(path))
    margin_col = find_column(df.columns, "dc_available_margin", source=str(path))

    if local_col and medium_col and remote_col:
        result = pd.DataFrame(
            {
                "time_step": to_numeric(df[time_col]),
                "datacenter_id": to_numeric(df[dc_col]),
                "local_allocation": to_numeric(df[local_col], fill_value=0),
                "medium_allocation": to_numeric(df[medium_col], fill_value=0),
                "remote_allocation": to_numeric(df[remote_col], fill_value=0),
            }
        )
        if margin_col:
            result["dc_available_margin"] = to_numeric(df[margin_col], fill_value=np.nan)
        else:
            result["dc_available_margin"] = np.nan
        result["total_allocation"] = result[
            ["local_allocation", "medium_allocation", "remote_allocation"]
        ].sum(axis=1)
        result = result.dropna(subset=["time_step", "datacenter_id"])
        if result["total_allocation"].sum() > 0:
            print(f"[读取] 使用真实 DC-区域宽表分配矩阵：{path}")
            return result

    region_col = find_column(df.columns, "region", source=str(path))
    allocation_col = find_column(df.columns, "allocation", source=str(path))
    if region_col is None or allocation_col is None:
        return None

    long_df = pd.DataFrame(
        {
            "time_step": to_numeric(df[time_col]),
            "datacenter_id": to_numeric(df[dc_col]),
            "region": df[region_col].map(map_region),
            "allocation": to_numeric(df[allocation_col], fill_value=0),
        }
    ).dropna(subset=["time_step", "datacenter_id", "region"])
    if long_df.empty:
        return None

    pivot = long_df.pivot_table(
        index=["time_step", "datacenter_id"],
        columns="region",
        values="allocation",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for region in REGIONS:
        if region not in pivot.columns:
            pivot[region] = 0.0
    result = pivot.rename(
        columns={
            "本地": "local_allocation",
            "中距离": "medium_allocation",
            "远距离": "remote_allocation",
        }
    )
    result["dc_available_margin"] = np.nan
    result["total_allocation"] = result[
        ["local_allocation", "medium_allocation", "remote_allocation"]
    ].sum(axis=1)
    if result["total_allocation"].sum() > 0:
        print(f"[读取] 使用真实 DC-区域长表分配矩阵：{path}")
        return result
    return None


def load_real_allocation_matrix(root: Path, output_matrix: Path, strategy: str) -> tuple[pd.DataFrame | None, Path | None]:
    """加载可用于动画展示的数据中心到区域分配矩阵。"""
    candidates = discover_candidate_files(root, output_matrix)
    print("[检查] DC-区域分配矩阵候选文件：")
    for path in candidates[:20]:
        print(f"  - {path}")
    if len(candidates) > 20:
        print(f"  其余候选文件数量：{len(candidates) - 20}")

    for path in candidates:
        if path.suffix.lower() != ".csv":
            continue
        matrix = try_load_real_matrix_from_csv(path, strategy)
        if matrix is not None:
            return matrix, path
    return None, None


def load_hourly_token_context(root: Path, strategy: str) -> tuple[pd.DataFrame, Path | None]:
    """读取小时级 Token 出口上下文数据。"""
    candidates = [
        root / "outputs" / "token_export" / "hourly_token_export.csv",
        root / "outputs" / "data" / "token_export" / "hourly_token_export.csv",
        root / "outputs" / "nbsdc_fusion" / "aligned_hourly_fusion.csv",
        root / "outputs" / "data" / "nbsdc_fusion" / "aligned_hourly_fusion.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        strategy_col = find_column(df.columns, "strategy", source=str(path))
        if strategy_col is not None and strategy in set(df[strategy_col].astype(str)):
            df = df[df[strategy_col].astype(str) == strategy].copy()
        time_col = find_column(df.columns, "time_step", source=str(path))
        margin_col = find_column(df.columns, "power_margin", source=str(path))
        if time_col is None or margin_col is None:
            continue
        result = pd.DataFrame(
            {
                "time_step": to_numeric(df[time_col]),
                "equivalent_power_margin": to_numeric(df[margin_col], fill_value=0.4),
            }
        )
        total_col = find_column(df.columns, "total_service", source=str(path))
        if total_col:
            result["reference_total_service"] = to_numeric(df[total_col], fill_value=np.nan)
        token_capacity_col = find_column(df.columns, "allocation", source=str(path))
        if token_capacity_col and "reference_total_service" not in result.columns:
            result["reference_total_service"] = to_numeric(df[token_capacity_col], fill_value=np.nan)
        latency_col = find_column(df.columns, "latency", source=str(path))
        sla_col = find_column(df.columns, "sla_violation", source=str(path))
        if latency_col:
            result["source_avg_latency"] = to_numeric(df[latency_col], fill_value=np.nan)
        if sla_col:
            result["source_sla_violation"] = to_numeric(df[sla_col], fill_value=np.nan)
        result = result.dropna(subset=["time_step"]).sort_values("time_step").reset_index(drop=True)
        if not result.empty:
            return result, path

    times = np.arange(24)
    result = pd.DataFrame(
        {
            "time_step": times,
            "equivalent_power_margin": 0.45 + 0.2 * np.sin(times / 24 * 2 * np.pi),
            "reference_total_service": np.nan,
        }
    )
    return result, None


def dc_margins_from_global(time_step: float, global_margin: float) -> np.ndarray:
    """根据全局功率裕度构造各数据中心的相对裕度。"""
    phases = np.array([0.0, 1.45, 2.9, 4.35])
    variations = 0.92 + 0.18 * np.sin(time_step / 3.2 + phases) + 0.07 * np.cos(time_step / 5.1 + phases)
    return np.clip(global_margin * variations, 0.05, 0.95)


def bounded_region_targets(
    total_capacity: float,
    time_step: float,
) -> np.ndarray:
    """生成满足上下限约束的地区服务需求目标。"""
    demand = total_capacity * np.array(
        [
            0.25 + 0.10 * math.sin(time_step / 4.0),
            0.40 + 0.10 * math.sin(time_step / 5.2 + 0.8),
            0.20 + 0.08 * math.cos(time_step / 4.6 + 0.3),
        ]
    )
    demand = np.clip(demand, 0, None)
    total_service = min(total_capacity * 0.92, float(demand.sum()))
    if total_service <= 0:
        return np.zeros(3)

    revenue = np.array([0.88, 1.05, 1.18])
    latency_penalty = np.array([0.06, 0.20, 0.36])
    sla_penalty = np.array([0.02, 0.10, 0.25])
    score = np.clip(revenue - 0.55 * latency_penalty - 0.45 * sla_penalty, 0.05, None)
    raw = demand * score
    target = raw / raw.sum() * total_service if raw.sum() > 0 else demand / demand.sum() * total_service

    lower_share = np.array([0.15, 0.30, 0.05])
    upper_share = np.array([0.45, 0.65, 0.35])
    lower = lower_share * total_service
    upper = upper_share * total_service

    # 如果某区域需求低于下限，放宽该区域下限，避免违反需求上限。
    lower = np.minimum(lower, demand)
    upper = np.minimum(upper, demand)
    target = np.clip(target, lower, upper)

    for _ in range(10):
        diff = total_service - float(target.sum())
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            room = np.maximum(upper - target, 0)
            if room.sum() <= 0:
                break
            target += diff * room / room.sum()
        else:
            removable = np.maximum(target - lower, 0)
            if removable.sum() <= 0:
                break
            target += diff * removable / removable.sum()

    return np.clip(target, 0, demand)


def allocate_targets_to_dcs(dc_capacity: np.ndarray, region_targets: np.ndarray, time_step: float) -> np.ndarray:
    """将地区目标服务量按数据中心能力分配到各数据中心。"""
    allocation = np.zeros((len(DC_IDS), len(REGIONS)))
    remaining = dc_capacity.astype(float).copy()
    # 行为偏好只影响可视化分布，不突破容量/需求约束。
    affinity = np.array(
        [
            [1.10, 0.95, 0.85],
            [0.90, 1.12, 0.92],
            [0.86, 1.00, 1.14],
            [1.00, 0.92, 1.05],
        ],
        dtype=float,
    )
    affinity *= 1.0 + 0.08 * np.sin(time_step / 4.0 + np.arange(4)[:, None] + np.arange(3)[None, :] * 0.7)
    priority = np.argsort(-region_targets)
    for region_idx in priority:
        target = float(region_targets[region_idx])
        if target <= 0:
            continue
        weights = remaining * np.clip(affinity[:, region_idx], 0.05, None)
        if weights.sum() <= 0:
            continue
        proposal = target * weights / weights.sum()
        placed = np.minimum(proposal, remaining)
        leftover = target - placed.sum()
        while leftover > 1e-6 and remaining.sum() - placed.sum() > 1e-6:
            spare = np.maximum(remaining - placed, 0)
            if spare.sum() <= 0:
                break
            add = np.minimum(leftover * spare / spare.sum(), spare)
            placed += add
            leftover -= add.sum()
        allocation[:, region_idx] = placed
        remaining -= placed
    return allocation


def construct_visualization_matrix(root: Path, strategy: str) -> tuple[pd.DataFrame, Path | None]:
    # 该分配矩阵为基于等效功率裕度构造的可视化场景，不代表真实跨区域算力交易数据。
    """基于实验输出构造动画使用的数据中心到区域服务矩阵。"""
    print("该分配矩阵为基于等效功率裕度构造的可视化场景，不代表真实跨区域算力交易数据。")
    context, source = load_hourly_token_context(root, strategy)
    records: list[dict[str, float]] = []

    for _, row in context.iterrows():
        time_step = float(row["time_step"])
        global_margin = float(np.clip(row["equivalent_power_margin"], 0.05, 0.95))
        dc_margins = dc_margins_from_global(time_step, global_margin)
        reference_total = finite_number(row.get("reference_total_service"))
        total_service_cap = max(reference_total if reference_total and reference_total > 0 else global_margin * 850_000, 80_000)
        dc_capacity = total_service_cap * dc_margins / dc_margins.sum()
        region_targets = bounded_region_targets(float(dc_capacity.sum()), time_step)
        matrix = allocate_targets_to_dcs(dc_capacity, region_targets, time_step)

        for dc_index, dc_id in enumerate(DC_IDS):
            local, medium, remote = matrix[dc_index, :]
            records.append(
                {
                    "time_step": time_step,
                    "datacenter_id": dc_id,
                    "local_allocation": local,
                    "medium_allocation": medium,
                    "remote_allocation": remote,
                    "total_allocation": local + medium + remote,
                    "dc_available_margin": dc_margins[dc_index],
                }
            )

    return pd.DataFrame(records), source


def complete_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """补齐分配矩阵中缺失的数据中心、区域和时间步组合。"""
    records: list[dict[str, float]] = []
    for time_step in sorted(matrix["time_step"].dropna().unique()):
        frame = matrix[matrix["time_step"] == time_step]
        for dc_id in DC_IDS:
            row = frame[frame["datacenter_id"] == dc_id]
            if row.empty:
                records.append(
                    {
                        "time_step": time_step,
                        "datacenter_id": dc_id,
                        "local_allocation": 0.0,
                        "medium_allocation": 0.0,
                        "remote_allocation": 0.0,
                        "total_allocation": 0.0,
                        "dc_available_margin": np.nan,
                    }
                )
            else:
                records.append(row.iloc[0].to_dict())
    result = pd.DataFrame(records)
    result["total_allocation"] = result[
        ["local_allocation", "medium_allocation", "remote_allocation"]
    ].sum(axis=1)
    if result["dc_available_margin"].isna().any():
        totals = result.groupby("time_step")["total_allocation"].transform("sum")
        result["dc_available_margin"] = result["dc_available_margin"].fillna(
            (result["total_allocation"] / totals.replace(0, np.nan)).fillna(0.25).clip(0.05, 0.95)
        )
    return result


def build_frame_metrics(matrix: pd.DataFrame) -> pd.DataFrame:
    """为动画每一帧计算汇总指标和展示文本。"""
    records = []
    for time_step, frame in matrix.groupby("time_step"):
        region_totals = frame[["local_allocation", "medium_allocation", "remote_allocation"]].sum()
        region_vector = region_totals.to_numpy(dtype=float)
        total = float(region_vector.sum())
        if total > 0:
            avg_latency = float(np.dot(region_vector, REGION_LATENCY_HOURS) / total)
            sla_rate = float(np.dot(region_vector, REGION_SLA_RISK) / total)
        else:
            avg_latency = 0.0
            sla_rate = 0.0
        records.append(
            {
                "time_step": time_step,
                "local_total": region_totals["local_allocation"],
                "medium_total": region_totals["medium_allocation"],
                "remote_total": region_totals["remote_allocation"],
                "total_allocation": total,
                "avg_latency": avg_latency,
                "sla_violation_rate": sla_rate,
                "avg_equivalent_power_margin": float(frame["dc_available_margin"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values("time_step").reset_index(drop=True)


def load_or_construct_matrix(root: Path, output_paths: OutputPaths, strategy: str) -> tuple[pd.DataFrame, str]:
    """优先加载已有矩阵，缺失时构造可视化矩阵。"""
    real_matrix, source = load_real_allocation_matrix(root, output_paths.matrix, strategy)
    if real_matrix is None:
        matrix, source = construct_visualization_matrix(root, strategy)
        source_text = str(source) if source is not None else "constructed_power_margin_scenario"
    else:
        matrix = real_matrix
        source_text = str(source)
    matrix = complete_matrix(matrix).sort_values(["time_step", "datacenter_id"]).reset_index(drop=True)
    output_paths.matrix.parent.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "time_step",
        "datacenter_id",
        "local_allocation",
        "medium_allocation",
        "remote_allocation",
        "total_allocation",
        "dc_available_margin",
    ]
    export_matrix = matrix[export_columns].copy()
    allocation_columns = ["local_allocation", "medium_allocation", "remote_allocation", "total_allocation"]
    # 动画内部使用原始 Token 计算线宽，CSV 按图中口径统一导出为 M Token。
    export_matrix[allocation_columns] = export_matrix[allocation_columns] / TOKEN_UNIT_SCALE
    export_matrix.to_csv(output_paths.matrix, index=False, encoding="utf-8-sig")
    print(f"[完成] DC-区域分配矩阵已导出：{output_paths.matrix}")
    print("[单位] CSV 中 allocation 字段单位为 M Token；dc_available_margin 为0~1归一化裕度。")
    return matrix, source_text


def choose_frames(times: list[float], frame_step: int, max_frames: int) -> list[float]:
    """根据最大帧数限制选择动画展示的时间步。"""
    if frame_step < 1:
        raise ValueError("--frame-step 必须 >= 1。")
    sampled = times[::frame_step]
    if max_frames > 0 and len(sampled) > max_frames:
        indices = np.linspace(0, len(sampled) - 1, max_frames)
        sampled = [sampled[int(round(index))] for index in indices]
    if not sampled:
        raise ValueError("[数据错误] 没有可用于动画的时间步。")
    print(f"[抽帧] 原始时间步 {len(times)} 个，导出帧数 {len(sampled)}。")
    return sampled


def setup_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """初始化动画画布、坐标轴和固定布局元素。"""
    fig = plt.figure(figsize=(14, 7.8), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.55, 1.15],
        height_ratios=[2.05, 1.55],
        wspace=0.12,
        hspace=0.18,
    )
    ax_net = fig.add_subplot(grid[:, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    ax_trend = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.035, right=0.975, top=0.925, bottom=0.075)
    fig.suptitle(TITLE, fontsize=18, fontweight="bold", color="#0b3a67", y=0.975)
    return fig, ax_net, ax_info, ax_trend


def draw_network(ax: plt.Axes, frame: pd.DataFrame, metrics_row: pd.Series, max_flow: float, max_region: float) -> None:
    """绘制当前帧的数据中心到区域算力服务网络。"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.04, 0.99)
    ax.axis("off")

    left_panel = FancyBboxPatch(
        (0.085, 0.145),
        0.205,
        0.755,
        boxstyle="round,pad=0.014,rounding_size=0.028",
        facecolor="#f6faff",
        edgecolor="#c9ddf2",
        linewidth=1.0,
        alpha=0.84,
    )
    right_panel = FancyBboxPatch(
        (0.735, 0.16),
        0.205,
        0.705,
        boxstyle="round,pad=0.014,rounding_size=0.028",
        facecolor="#f8fcfa",
        edgecolor="#cbe8d5",
        linewidth=1.0,
        alpha=0.84,
    )
    ax.add_patch(left_panel)
    ax.add_patch(right_panel)
    ax.text(0.187, 0.875, "数据中心侧", ha="center", va="center", fontsize=11.6, color="#0b3a67", fontweight="bold")
    ax.text(0.837, 0.835, "服务区域侧", ha="center", va="center", fontsize=11.6, color="#1b5e3a", fontweight="bold")

    dc_positions = {
        0: (0.187, 0.775),
        1: (0.187, 0.595),
        2: (0.187, 0.415),
        3: (0.187, 0.235),
    }
    region_positions = {
        "本地": (0.837, 0.725),
        "中距离": (0.837, 0.505),
        "远距离": (0.837, 0.285),
    }
    region_colors = {"本地": "#2f80ed", "中距离": "#159a83", "远距离": "#0b3a67"}

    frame_by_dc = frame.set_index("datacenter_id")
    region_totals = {
        "本地": float(metrics_row["local_total"]),
        "中距离": float(metrics_row["medium_total"]),
        "远距离": float(metrics_row["remote_total"]),
    }

    for dc_id in DC_IDS:
        row = frame_by_dc.loc[dc_id]
        start = dc_positions[dc_id]
        values = {
            "本地": float(row["local_allocation"]),
            "中距离": float(row["medium_allocation"]),
            "远距离": float(row["remote_allocation"]),
        }
        for region_index, region in enumerate(REGIONS):
            value = values[region]
            if value <= 0:
                continue
            ratio = 0 if max_flow <= 0 else math.sqrt(value / max_flow)
            width = min(4.5, 0.8 + 3.7 * ratio)
            alpha = min(0.65, 0.35 + 0.30 * ratio)
            # 曲率按 DC 和区域共同变化，减轻 12 条线的交叉遮挡。
            rad = (region_index - 1) * 0.105 - (dc_id - 1.5) * 0.026
            arrow = FancyArrowPatch(
                start,
                region_positions[region],
                arrowstyle="-|>",
                mutation_scale=7.5,
                linewidth=width,
                color=region_colors[region],
                alpha=alpha,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=18,
                shrinkB=24,
                zorder=1,
            )
            ax.add_patch(arrow)

    for dc_id in DC_IDS:
        row = frame_by_dc.loc[dc_id]
        margin = float(row["dc_available_margin"])
        pos = dc_positions[dc_id]
        size = 600 + 1700 * np.clip(margin, 0, 1)
        ax.scatter([pos[0]], [pos[1]], s=size, color="#edf7ff", edgecolor="#2a72b5", linewidth=1.65, zorder=3)
        ax.text(pos[0], pos[1] + 0.004, f"DC{dc_id}", ha="center", va="center", fontsize=10.6, color="#0b3a67", fontweight="bold", zorder=4)
        ax.text(pos[0], pos[1] - 0.064, f"可用 {margin * 100:.1f}%", ha="center", va="center", fontsize=8.0, color="#49657f", zorder=4)

    for region in REGIONS:
        value = region_totals[region]
        pos = region_positions[region]
        ratio = 0 if max_region <= 0 else value / max_region
        size = 720 + 2200 * math.sqrt(max(ratio, 0))
        ax.scatter([pos[0]], [pos[1]], s=size, color="#fbfefd", edgecolor=region_colors[region], linewidth=2.0, zorder=3)
        ax.text(pos[0], pos[1] + 0.010, region, ha="center", va="center", fontsize=11.2, color=region_colors[region], fontweight="bold", zorder=4)
        ax.text(pos[0], pos[1] - 0.064, format_m_token(value), ha="center", va="center", fontsize=8.0, color="#17324d", zorder=4)

    ax.text(
        0.50,
        0.090,
        "节点大小表示裕度/服务量，连线粗细表示分配强度",
        ha="center",
        va="center",
        fontsize=9.7,
        color="#49657f",
    )


def draw_info(ax: plt.Axes, metrics_row: pd.Series) -> None:
    """绘制当前帧的指标信息面板。"""
    ax.clear()
    ax.axis("off")
    lines = [
        f"当前时间步：{int(metrics_row['time_step']) if float(metrics_row['time_step']).is_integer() else metrics_row['time_step']}",
        f"总分配量：{format_m_token(metrics_row['total_allocation'])}",
        f"本地分配量：{format_m_token(metrics_row['local_total'])}",
        f"中距离分配量：{format_m_token(metrics_row['medium_total'])}",
        f"远距离分配量：{format_m_token(metrics_row['remote_total'])}",
        f"平均时延：{format_float(metrics_row['avg_latency'], 1, ' h')}",
        f"SLA违约率：{format_percent(metrics_row['sla_violation_rate'])}",
        f"平均等效功率裕度：{format_percent(metrics_row['avg_equivalent_power_margin'])}",
    ]
    ax.text(
        0.045,
        0.91,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="#17324d",
        linespacing=1.32,
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#f2f8ff",
            "edgecolor": "#b9d7f0",
            "linewidth": 1.2,
        },
    )


def draw_trend(ax: plt.Axes, metrics: pd.DataFrame, time_step: float) -> None:
    """绘制当前帧之前的时序趋势曲线。"""
    ax.clear()
    x = metrics["time_step"].to_numpy(dtype=float)
    local = metrics["local_total"].to_numpy(dtype=float) / TOKEN_UNIT_SCALE
    medium = metrics["medium_total"].to_numpy(dtype=float) / TOKEN_UNIT_SCALE
    remote = metrics["remote_total"].to_numpy(dtype=float) / TOKEN_UNIT_SCALE
    colors = ["#2f80ed", "#159a83", "#0b3a67"]
    ax.plot(x, local, color=colors[0], linewidth=1.9, label="本地")
    ax.plot(x, medium, color=colors[1], linewidth=1.9, label="中距离")
    ax.plot(x, remote, color=colors[2], linewidth=1.9, label="远距离")
    ax.axvline(time_step, color="#0b3a67", linewidth=1.45, alpha=0.90)
    ax.set_title(TREND_TITLE, fontsize=10.2, color="#17324d")
    ax.set_xlabel("时间/h", fontsize=9)
    ax.set_ylabel(f"服务量（{TOKEN_UNIT_LABEL}）", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(labelsize=8)
    max_service_value = float(np.nanmax(np.vstack([local, medium, remote])))
    ymax = max_service_value * 1.2 if max_service_value > 0 else 1.0
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.78, edgecolor="#d6e4f2")


def create_animation(matrix: pd.DataFrame, metrics: pd.DataFrame, frames: list[float], fps: int) -> animation.FuncAnimation:
    """创建 Matplotlib 动画对象并绑定帧更新逻辑。"""
    fig, ax_net, ax_info, ax_trend = setup_axes()
    matrix_by_time = {time: frame for time, frame in matrix.groupby("time_step")}
    metrics_by_time = metrics.set_index("time_step", drop=False)
    max_flow = float(
        matrix[["local_allocation", "medium_allocation", "remote_allocation"]].to_numpy().max()
    )
    max_region = float(metrics[["local_total", "medium_total", "remote_total"]].to_numpy().max())

    def update(time_step: float):
        """根据当前帧刷新动画中的图层和指标展示。"""
        frame = matrix_by_time[time_step]
        metrics_row = metrics_by_time.loc[time_step]
        if isinstance(metrics_row, pd.DataFrame):
            metrics_row = metrics_row.iloc[0]
        draw_network(ax_net, frame, metrics_row, max_flow, max_region)
        draw_info(ax_info, metrics_row)
        draw_trend(ax_trend, metrics, time_step)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=int(1000 / max(fps, 1)),
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )
    anim._codex_figure = fig
    return anim


def save_cover_image(
    matrix: pd.DataFrame,
    metrics: pd.DataFrame,
    time_step: float,
    output_paths: OutputPaths,
    *,
    dpi: int,
) -> None:
    """保存动画首帧或指定帧作为封面图。"""
    fig, ax_net, ax_info, ax_trend = setup_axes()
    try:
        frame = matrix[matrix["time_step"] == time_step]
        metrics_row = metrics[metrics["time_step"] == time_step]
        if metrics_row.empty:
            raise ValueError(f"[数据错误] 找不到封面帧对应的时间步：{time_step}")
        max_flow = float(
            matrix[["local_allocation", "medium_allocation", "remote_allocation"]].to_numpy().max()
        )
        max_region = float(metrics[["local_total", "medium_total", "remote_total"]].to_numpy().max())
        draw_network(ax_net, frame, metrics_row.iloc[0], max_flow, max_region)
        draw_info(ax_info, metrics_row.iloc[0])
        draw_trend(ax_trend, metrics, time_step)

        output_paths.cover.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_paths.cover, dpi=dpi, facecolor="white")
        print(f"[完成] 封面图已生成：{output_paths.cover}")
    finally:
        plt.close(fig)


def save_animation(
    anim: animation.FuncAnimation,
    frame_count: int,
    output_paths: OutputPaths,
    *,
    fps: int,
    dpi: int,
    skip_mp4: bool,
) -> None:
    """导出 GIF 或 MP4 动画文件。"""
    output_paths.gif.parent.mkdir(parents=True, exist_ok=True)
    gif_writer = animation.PillowWriter(fps=fps)
    with tqdm(total=frame_count, desc="导出 GIF") as progress:
        def gif_progress(current_frame: int, total_frames: int) -> None:
            """接收 GIF 导出进度并更新进度条。"""
            if total_frames and progress.total != total_frames:
                progress.total = total_frames
            progress.update(max(0, current_frame + 1 - progress.n))

        anim.save(output_paths.gif, writer=gif_writer, dpi=dpi, progress_callback=gif_progress)
    print(f"[完成] GIF 已生成：{output_paths.gif}")

    if skip_mp4:
        print("[提示] 已按 --skip-mp4 跳过 MP4 导出。")
        return

    if not animation.writers.is_available("ffmpeg"):
        print("MP4 导出失败，可能是未安装 FFmpeg；GIF 已正常生成。")
        return

    try:
        mp4_writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        with tqdm(total=frame_count, desc="导出 MP4") as progress:
            def mp4_progress(current_frame: int, total_frames: int) -> None:
                """接收 MP4 导出进度并更新进度条。"""
                if total_frames and progress.total != total_frames:
                    progress.total = total_frames
                progress.update(max(0, current_frame + 1 - progress.n))

            anim.save(output_paths.mp4, writer=mp4_writer, dpi=dpi, progress_callback=mp4_progress)
        print(f"[完成] MP4 已生成：{output_paths.mp4}")
    except Exception as exc:
        print(f"MP4 导出失败，可能是未安装 FFmpeg；GIF 已正常生成。错误信息：{exc}")


# 答辩说明：
# 该动图以数据中心为供给侧节点，以本地、中距离、远距离为服务区域节点，构造并展示
# 每个时间步的数据中心—区域算力服务分配矩阵。节点大小表示可用裕度或区域服务量，
# 连线粗细表示对应数据中心向服务区域分配的算力服务量，用于直观展示跨域算力服务
# 在时间维度和空间维度上的分布变化。


def main() -> int:
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    args = parse_args()
    configure_chinese_font()

    root = Path.cwd()
    output_dir = args.output_dir.resolve()
    output_paths = OutputPaths(
        gif=output_dir / OUTPUT_GIF_NAME,
        mp4=output_dir / OUTPUT_MP4_NAME,
        cover=output_dir / OUTPUT_COVER_NAME,
        matrix=output_dir / OUTPUT_MATRIX_NAME,
    )

    matrix, source_text = load_or_construct_matrix(root, output_paths, args.strategy)
    metrics = build_frame_metrics(matrix)
    print(f"[数据] 使用来源：{source_text}")
    print(
        "[数据] 分配矩阵规模："
        f"{matrix['time_step'].nunique()} 个时间步，"
        f"{matrix['datacenter_id'].nunique()} 个数据中心，"
        f"总分配量范围 "
        f"{metrics['total_allocation'].min() / TOKEN_UNIT_SCALE:.2f}~"
        f"{metrics['total_allocation'].max() / TOKEN_UNIT_SCALE:.2f} {TOKEN_UNIT_LABEL}"
    )

    times = sorted(float(value) for value in matrix["time_step"].dropna().unique())
    frames = choose_frames(times, args.frame_step, args.max_frames)
    save_cover_image(matrix, metrics, frames[0], output_paths, dpi=args.dpi)
    anim = create_animation(matrix, metrics, frames, args.fps)
    try:
        save_animation(
            anim,
            len(frames),
            output_paths,
            fps=args.fps,
            dpi=args.dpi,
            skip_mp4=args.skip_mp4,
        )
    finally:
        fig = getattr(anim, "_codex_figure", None)
        if fig is not None:
            plt.close(fig)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\n[错误] {error}", file=sys.stderr)
        raise SystemExit(1)
