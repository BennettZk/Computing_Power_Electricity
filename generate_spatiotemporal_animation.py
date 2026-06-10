# -*- coding: utf-8 -*-
"""
生成“功率约束下数据中心负载时空演化过程”GIF/MP4。

依赖安装命令：
    pip install pandas matplotlib openpyxl imageio pillow tqdm

说明：
    1. 主图为“数据中心编号 × 机房编号”的负载热力图，不使用真实地理地图或经纬度。
    2. 默认每个时间步取一帧，可通过 --frame-step 调整为 2、5、10 等以减少帧数。
    3. 如本机可用 ffmpeg，会额外导出 MP4；否则只导出 GIF 并给出提示。
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from openpyxl import load_workbook
from tqdm import tqdm


TITLE = "功率约束下数据中心负载时空演化过程"
HEATMAP_TITLE = "数据中心负载率时空演化热力图"

SERVER_FILE_NAME = "服务器级别的调度数据.xlsx"
CLUSTER_FILE_NAME = "数据中心集群级别的调度数据.xlsx"
CHIP_FILE_NAME = "芯片级别的调度数据.xlsx"

FRAME_STEP = 1
MAX_FRAMES = 0
FPS = 8
DPI = 140
DEFAULT_SERVER_CPU_CORES = 64
DEFAULT_RACKS_PER_ROOM = 10
DEFAULT_SERVERS_PER_RACK = 10
DEFAULT_ROOM_CPU_CAPACITY = (
    DEFAULT_RACKS_PER_ROOM * DEFAULT_SERVERS_PER_RACK * DEFAULT_SERVER_CPU_CORES
)

ROOM_SHEET_RE = re.compile(r"^DC\d+_R\d+$", re.IGNORECASE)
RACK_SHEET_RE = re.compile(r"^DC\d+_R\d+_K\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class OutputPaths:
    gif: Path
    mp4: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回脚本运行配置。"""
    parser = argparse.ArgumentParser(
        description="生成数据中心负载时空演化 GIF/MP4，用于论文答辩展示。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Excel 数据目录，默认 data/raw。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="输出目录，默认 output。",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=FRAME_STEP,
        help="抽帧间隔。1 表示每个时间步一帧，2 表示每隔 2 个时间步取一帧。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="最多导出多少帧；设为 0 表示不限制。",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="动画帧率，默认 8 FPS。")
    parser.add_argument("--dpi", type=int, default=DPI, help="导出分辨率，默认 140 DPI。")
    parser.add_argument(
        "--skip-mp4",
        action="store_true",
        help="只导出 GIF，不尝试导出 MP4。",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    """字段名标准化：处理中英文符号、空格、大小写差异。"""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[\s_\-./\\()（）\[\]【】{}:：,，;；]+", "", text)
    return text


FIELD_CANDIDATES: dict[str, list[str]] = {
    "time_step": [
        "time_step",
        "time",
        "timestamp",
        "t",
        "step",
        "时间步",
        "时间步(5分钟)",
        "scheduled start time",
    ],
    "datacenter_id": [
        "datacenter_id",
        "dc_id",
        "data_center",
        "dc",
        "datacenter",
        "数据中心编号",
        "数据中心id",
    ],
    "server_room_id": [
        "server_room_id",
        "room_id",
        "room",
        "server_room",
        "机房编号",
        "服务器机房编号",
        "机房id",
    ],
    "rack_id": ["rack_id", "rack", "机架编号", "机架id"],
    "server_id": ["server_id", "server", "服务器编号", "服务器id"],
    "cpu_usage": [
        "cpu_usage",
        "cpu_used",
        "used cpu",
        "used_cpu",
        "cpu",
        "cpu_utilization",
        "cpu使用量",
        "cpu用量",
        "cpu负载",
        "CPU",
    ],
    "cpu_capacity": [
        "cpu_capacity",
        "cpu_total",
        "total_cpu",
        "cpu_limit",
        "server_cpu_capacity",
        "capacity",
        "cpu容量",
        "cpu总量",
        "cpu上限",
    ],
    "task_count": [
        "task_count",
        "num_tasks",
        "tasks",
        "job_count",
        "processed_tasks",
        "任务数量",
        "任务数",
        "任务处理数量",
    ],
    "power_cap": [
        "power_cap",
        "power_limit",
        "cap",
        "normalized_power_cap",
        "Power Cap",
        "功率上限",
        "功率上限(标幺值)",
        "功率上限(瓦)",
    ],
    "electricity_price": [
        "electricity_price",
        "price",
        "power_price",
        "电价",
        "电价(美元/兆瓦时)",
    ],
    "actual_power": ["actual_power", "实际功率", "实际功率(瓦)"],
    "reported_power": ["reported_power", "报量功率", "报量功率(瓦)"],
}


def find_column(
    columns: Iterable[object],
    role: str,
    *,
    required: bool = False,
    source: str = "",
) -> str | None:
    """根据语义候选词自动识别字段名。"""
    original_columns = [str(col) for col in columns if col is not None and str(col) != "nan"]
    normalized = {normalize_name(col): col for col in original_columns}
    candidates = FIELD_CANDIDATES[role]

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    # 允许“时间步(5分钟)”匹配“时间步”等包含关系。
    for candidate in candidates:
        key = normalize_name(candidate)
        if not key or len(key) <= 3:
            continue
        for norm_col, original_col in normalized.items():
            if key in norm_col or norm_col in key:
                return original_col

    if required:
        raise ValueError(
            f"[字段缺失] {source} 缺少语义字段 '{role}'。"
            f"已有字段：{original_columns}"
        )
    return None


def to_numeric(series: pd.Series, *, fill_value: float | None = None) -> pd.Series:
    """将输入序列转换为数值序列，并用默认值处理异常项。"""
    result = pd.to_numeric(series.replace("", np.nan), errors="coerce")
    if fill_value is not None:
        result = result.fillna(fill_value)
    return result


def format_value(value: object, digits: int = 0, suffix: str = "") -> str:
    """格式化图表信息框中的数值。"""
    if value is None:
        return "无数据"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "无数据"
    if math.isnan(number):
        return "无数据"
    if digits == 0:
        return f"{number:,.0f}{suffix}"
    return f"{number:,.{digits}f}{suffix}"


def format_percent(value: object, digits: int = 1) -> str:
    """把比例值格式化为百分比文本。"""
    number = finite_number(value)
    if number is None:
        return "无数据"
    return f"{number * 100:.{digits}f}%"


def format_time_step(value: object) -> str:
    """把时间步格式化为可读的时间标签。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def finite_number(value: object) -> float | None:
    """判断输入值是否为有限数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def configure_chinese_font() -> None:
    """优先使用常见中文字体；若不可用则降级，不让脚本崩溃。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
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
    print("[字体提示] 未检测到常见中文字体，已降级为默认字体；图中文字可能无法完整显示。")


def find_excel_file(data_dir: Path, file_name: str, keywords: list[str]) -> Path:
    """在候选目录中查找符合关键词的 Excel 文件。"""
    exact_path = data_dir / file_name
    if exact_path.exists():
        return exact_path

    matches = [
        path
        for path in data_dir.rglob("*.xlsx")
        if all(keyword in path.name for keyword in keywords)
    ]
    if matches:
        print(f"[文件提示] 未找到精确文件名 {file_name}，使用匹配文件：{matches[0]}")
        return matches[0]

    existing = [str(path) for path in data_dir.rglob("*.xlsx")] if data_dir.exists() else []
    raise FileNotFoundError(
        f"[文件缺失] 未找到 {file_name}。请检查数据目录：{data_dir}\n"
        f"当前可见 xlsx 文件：{existing if existing else '无'}"
    )


def inspect_workbook(path: Path) -> None:
    """打印工作簿结构摘要，避免假设 sheet 和字段固定不变。"""
    print(f"\n[检查] {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_names = workbook.sheetnames
    room_sheets = [name for name in sheet_names if ROOM_SHEET_RE.fullmatch(name)]
    rack_sheets = [name for name in sheet_names if RACK_SHEET_RE.fullmatch(name)]
    print(f"  sheet 数量：{len(sheet_names)}")
    print(f"  前 15 个 sheet：{sheet_names[:15]}")
    if len(sheet_names) > 15:
        print(f"  其余 sheet 数量：{len(sheet_names) - 15}")
    if room_sheets or rack_sheets:
        print(f"  识别到机房级 sheet：{len(room_sheets)} 个；机架级 sheet：{len(rack_sheets)} 个")

    representative_names: list[str] = []
    for preferred in [
        "Sheet1",
        "step_cpu_usage_vs_cap",
        "datacenter",
        "output3_chip",
    ]:
        if preferred in sheet_names:
            representative_names.append(preferred)
    if room_sheets:
        representative_names.append(room_sheets[0])
    if rack_sheets:
        representative_names.append(rack_sheets[0])
    if not representative_names and sheet_names:
        representative_names.append(sheet_names[0])

    seen: set[str] = set()
    for sheet_name in representative_names:
        if sheet_name in seen:
            continue
        seen.add(sheet_name)
        sheet = workbook[sheet_name]
        first_row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        print(
            f"  示例 sheet={sheet_name!r}, rows={sheet.max_row}, "
            f"cols={sheet.max_column}, 字段={list(first_row)}"
        )
    workbook.close()


def standardize_spatial_frame(
    df: pd.DataFrame,
    *,
    source: str,
    include_rack: bool = False,
) -> pd.DataFrame:
    """字段识别与映射：将不同表头统一为标准列。"""
    time_col = find_column(df.columns, "time_step", required=True, source=source)
    dc_col = find_column(df.columns, "datacenter_id", required=True, source=source)
    room_col = find_column(df.columns, "server_room_id", required=True, source=source)
    rack_col = find_column(df.columns, "rack_id", required=include_rack, source=source)
    server_col = find_column(df.columns, "server_id", source=source)
    cpu_col = find_column(df.columns, "cpu_usage", required=True, source=source)
    capacity_col = find_column(df.columns, "cpu_capacity", source=source)
    task_col = find_column(df.columns, "task_count", source=source)

    result = pd.DataFrame(
        {
            "time_step": to_numeric(df[time_col]),
            "datacenter_id": to_numeric(df[dc_col]),
            "server_room_id": to_numeric(df[room_col]),
            "cpu_usage": to_numeric(df[cpu_col], fill_value=0),
        }
    )
    result["cpu_capacity"] = (
        to_numeric(df[capacity_col]) if capacity_col is not None else np.nan
    )
    result["task_count"] = to_numeric(df[task_col], fill_value=0) if task_col else np.nan
    if include_rack and rack_col:
        result["rack_id"] = to_numeric(df[rack_col])
    if include_rack and server_col:
        result["server_id"] = to_numeric(df[server_col])

    result = result.dropna(subset=["time_step", "datacenter_id", "server_room_id"])
    return result


def infer_room_capacity_from_rack_sheets(
    xls: pd.ExcelFile, rack_sheets: list[str]
) -> pd.DataFrame:
    """从机架级 sheet 统计每个机房的真实服务器数量，并换算 CPU 容量。"""
    if not rack_sheets:
        return pd.DataFrame(columns=["datacenter_id", "server_room_id", "inferred_cpu_capacity"])

    records: list[pd.DataFrame] = []
    skipped_examples: list[str] = []
    for sheet_name in tqdm(rack_sheets, desc="统计机房服务器数量"):
        sample = xls.parse(sheet_name, nrows=5)
        dc_col = find_column(sample.columns, "datacenter_id", source=sheet_name)
        room_col = find_column(sample.columns, "server_room_id", source=sheet_name)
        rack_col = find_column(sample.columns, "rack_id", source=sheet_name)
        server_col = find_column(sample.columns, "server_id", source=sheet_name)
        required_cols = [dc_col, room_col, rack_col, server_col]
        if any(col is None for col in required_cols):
            if len(skipped_examples) < 3:
                skipped_examples.append(f"{sheet_name}: {list(sample.columns)}")
            continue

        df = xls.parse(sheet_name, usecols=required_cols)
        dedup = pd.DataFrame(
            {
                "datacenter_id": to_numeric(df[dc_col]),
                "server_room_id": to_numeric(df[room_col]),
                "rack_id": to_numeric(df[rack_col]),
                "server_id": to_numeric(df[server_col]),
            }
        ).dropna()
        if not dedup.empty:
            records.append(dedup.drop_duplicates())

    if not records:
        if skipped_examples:
            print(
                "[容量提示] 无法从机架级 sheet 统计服务器数量，"
                f"示例缺失字段：{skipped_examples}"
            )
        return pd.DataFrame(columns=["datacenter_id", "server_room_id", "inferred_cpu_capacity"])

    servers = pd.concat(records, ignore_index=True).drop_duplicates()
    capacity = (
        servers.groupby(["datacenter_id", "server_room_id"], as_index=False)
        .size()
        .rename(columns={"size": "server_count"})
    )
    capacity["inferred_cpu_capacity"] = capacity["server_count"] * DEFAULT_SERVER_CPU_CORES
    print(
        "[容量] 已从机架级 server_id/rack_id 统计机房服务器数量："
        f"最小 {capacity['server_count'].min():.0f} 台，"
        f"最大 {capacity['server_count'].max():.0f} 台；"
        f"单服务器容量按 {DEFAULT_SERVER_CPU_CORES} 核计算。"
    )
    return capacity[["datacenter_id", "server_room_id", "inferred_cpu_capacity"]]


def apply_cpu_capacity_and_load_rate(
    agg: pd.DataFrame,
    capacity_by_room: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """计算 CPU 负载率，并把结果限制在 0~1。"""
    result = agg.copy()
    explicit_capacity = (
        "cpu_capacity" in result.columns
        and result["cpu_capacity"].notna().any()
        and (result["cpu_capacity"] > 0).any()
    )

    capacity_source_parts: list[str] = []
    if explicit_capacity:
        capacity_source_parts.append("显式 CPU 容量字段")
    else:
        result["cpu_capacity"] = np.nan

    if not capacity_by_room.empty:
        result = result.merge(capacity_by_room, on=["datacenter_id", "server_room_id"], how="left")
        missing_capacity = result["cpu_capacity"].isna() | (result["cpu_capacity"] <= 0)
        result.loc[missing_capacity, "cpu_capacity"] = result.loc[
            missing_capacity, "inferred_cpu_capacity"
        ]
        result = result.drop(columns=["inferred_cpu_capacity"])
        capacity_source_parts.append("机架级真实服务器数量 × 64 核")

    missing_capacity = result["cpu_capacity"].isna() | (result["cpu_capacity"] <= 0)
    if missing_capacity.any():
        result.loc[missing_capacity, "cpu_capacity"] = DEFAULT_ROOM_CPU_CAPACITY
        capacity_source_parts.append(
            f"默认机房容量 {DEFAULT_ROOM_CPU_CAPACITY} 核"
            f"({DEFAULT_RACKS_PER_ROOM}机架×{DEFAULT_SERVERS_PER_RACK}服务器×{DEFAULT_SERVER_CPU_CORES}核)"
        )

    raw_rate = result["cpu_usage"] / result["cpu_capacity"].replace(0, np.nan)
    result["load_rate"] = raw_rate.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    result["heat_value"] = result["load_rate"]
    capacity_source = "；".join(dict.fromkeys(capacity_source_parts))
    return result, capacity_source


def load_spatial_data(server_path: Path) -> pd.DataFrame:
    """数据读取与聚合：优先使用机房级 sheet，回退到机架级 sheet。"""
    xls = pd.ExcelFile(server_path, engine="openpyxl")
    room_sheets = [name for name in xls.sheet_names if ROOM_SHEET_RE.fullmatch(name)]
    rack_sheets = [name for name in xls.sheet_names if RACK_SHEET_RE.fullmatch(name)]

    if room_sheets:
        print(f"\n[读取] 使用服务器级文件中的 {len(room_sheets)} 个机房级 sheet：DCx_Ry")
        frames = []
        for sheet_name in tqdm(room_sheets, desc="读取机房级数据"):
            df = xls.parse(sheet_name)
            frames.append(
                standardize_spatial_frame(
                    df,
                    source=f"{server_path.name}/{sheet_name}",
                )
            )
    elif rack_sheets:
        print(
            f"\n[读取] 未找到机房级 sheet，回退到 {len(rack_sheets)} 个机架级 sheet：DCx_Ry_Kz"
        )
        frames = []
        for sheet_name in tqdm(rack_sheets, desc="读取机架级数据"):
            df = xls.parse(sheet_name)
            standardized = standardize_spatial_frame(
                df,
                source=f"{server_path.name}/{sheet_name}",
                include_rack=True,
            )
            frames.append(standardized)
    else:
        raise ValueError(
            f"[结构缺失] {server_path} 未找到 DCx_Ry 或 DCx_Ry_Kz 格式的 sheet。"
            f"已有 sheet：{xls.sheet_names}"
        )

    spatial = pd.concat(frames, ignore_index=True)
    agg = (
        spatial.groupby(["time_step", "datacenter_id", "server_room_id"], as_index=False)
        .agg(
            cpu_usage=("cpu_usage", "sum"),
            task_count=("task_count", "sum"),
            cpu_capacity=("cpu_capacity", lambda values: values.sum(min_count=1)),
        )
        .fillna({"cpu_usage": 0, "task_count": 0})
    )
    capacity_by_room = (
        infer_room_capacity_from_rack_sheets(xls, rack_sheets)
        if not (agg["cpu_capacity"].notna().any() and (agg["cpu_capacity"] > 0).any())
        else pd.DataFrame(columns=["datacenter_id", "server_room_id", "inferred_cpu_capacity"])
    )
    agg, capacity_source = apply_cpu_capacity_and_load_rate(agg, capacity_by_room)
    print(
        "[聚合] 空间负载率数据："
        f"{agg['time_step'].nunique()} 个时间步，"
        f"{agg['datacenter_id'].nunique()} 个数据中心，"
        f"{agg['server_room_id'].nunique()} 个机房，"
        f"容量来源={capacity_source}"
    )
    print(
        "[聚合] CPU 负载率范围："
        f"平均 {agg['load_rate'].mean() * 100:.1f}%，"
        f"最大 {agg['load_rate'].max() * 100:.1f}%"
    )
    return agg


def load_cluster_metadata(cluster_path: Path) -> pd.DataFrame:
    """读取集群级辅助信息：时间步、功率上限、电价。"""
    xls = pd.ExcelFile(cluster_path, engine="openpyxl")
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        time_col = find_column(df.columns, "time_step", source=f"{cluster_path.name}/{sheet_name}")
        if time_col is None:
            continue
        power_col = find_column(df.columns, "power_cap", source=f"{cluster_path.name}/{sheet_name}")
        price_col = find_column(
            df.columns, "electricity_price", source=f"{cluster_path.name}/{sheet_name}"
        )
        if power_col is None and price_col is None:
            continue
        result = pd.DataFrame({"time_step": to_numeric(df[time_col])})
        if power_col:
            result["cluster_power_cap"] = to_numeric(df[power_col])
        if price_col:
            result["electricity_price"] = to_numeric(df[price_col])
        result = result.dropna(subset=["time_step"])
        print(f"[读取] 集群级辅助信息：{cluster_path.name}/{sheet_name}, 字段={list(df.columns)}")
        return result

    print(f"[提示] 集群级文件未找到可用时间步/功率/电价字段：{cluster_path}")
    return pd.DataFrame(columns=["time_step"])


def load_server_step_metadata(server_path: Path) -> pd.DataFrame:
    """读取服务器级全局 CPU 与约束数据（如存在 step_cpu_usage_vs_cap）。"""
    xls = pd.ExcelFile(server_path, engine="openpyxl")
    if "step_cpu_usage_vs_cap" not in xls.sheet_names:
        print("[提示] 服务器级文件未找到 step_cpu_usage_vs_cap sheet。")
        return pd.DataFrame(columns=["time_step"])

    sheet_name = "step_cpu_usage_vs_cap"
    df = xls.parse(sheet_name)
    time_col = find_column(df.columns, "time_step", required=True, source=sheet_name)
    used_cpu_col = find_column(df.columns, "cpu_usage", source=sheet_name)
    power_col = find_column(df.columns, "power_cap", source=sheet_name)

    result = pd.DataFrame({"time_step": to_numeric(df[time_col])})
    if used_cpu_col:
        result["server_used_cpu"] = to_numeric(df[used_cpu_col])
    if power_col:
        result["server_power_cap"] = to_numeric(df[power_col])
    result = result.dropna(subset=["time_step"])
    print(f"[读取] 服务器级全局约束信息：{server_path.name}/{sheet_name}, 字段={list(df.columns)}")
    return result


def load_chip_metadata(chip_path: Path) -> pd.DataFrame:
    """读取芯片级功率数据，仅在可按 time_step 可靠对齐时返回归一化指标。"""
    xls = pd.ExcelFile(chip_path, engine="openpyxl")
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        actual_col = find_column(df.columns, "actual_power", source=f"{chip_path.name}/{sheet_name}")
        cap_col = find_column(df.columns, "power_cap", source=f"{chip_path.name}/{sheet_name}")
        if actual_col is None or cap_col is None:
            continue

        time_col = find_column(df.columns, "time_step", source=f"{chip_path.name}/{sheet_name}")
        if time_col is None:
            print(
                "[提示] 芯片级文件未发现显式 time_step 字段，无法可靠对齐到调度时间步；"
                "右侧信息框不展示归一化实际功率和功率利用率。"
            )
            return pd.DataFrame(columns=["time_step"])

        result_values = pd.DataFrame(
            {
                "time_step": to_numeric(df[time_col]),
                "chip_actual_power": to_numeric(df[actual_col]),
                "chip_power_cap": to_numeric(df[cap_col]),
            }
        ).dropna(subset=["time_step", "chip_actual_power", "chip_power_cap"])
        if result_values.empty:
            return pd.DataFrame(columns=["time_step"])

        grouped = result_values.groupby("time_step", as_index=False).mean(numeric_only=True)
        cap = grouped["chip_power_cap"].replace(0, np.nan)
        grouped["normalized_actual_power"] = grouped["chip_actual_power"] / cap
        grouped["power_utilization"] = grouped["normalized_actual_power"]
        print(
            f"[读取] 芯片级归一化功率信息：{chip_path.name}/{sheet_name}，"
            "实际功率已按同一时间步芯片功率上限归一化。"
        )
        return grouped[["time_step", "normalized_actual_power", "power_utilization"]]

    print(f"[提示] 芯片级文件未找到可归一化的实际功率/功率上限字段：{chip_path}")
    return pd.DataFrame(columns=["time_step"])


def choose_sampled_times(
    all_time_steps: list[float], frame_step: int, max_frames: int
) -> list[float]:
    """从完整时间序列中选择用于动画展示的采样时间步。"""
    if frame_step < 1:
        raise ValueError("--frame-step 必须 >= 1。")
    sampled = all_time_steps[::frame_step]
    if max_frames > 0 and len(sampled) > max_frames:
        indices = np.linspace(0, len(sampled) - 1, max_frames)
        sampled = [sampled[int(round(index))] for index in indices]
    if not sampled:
        raise ValueError("[数据错误] 没有可用于动画的时间步。")
    print(
        f"[抽帧] 原始时间步 {len(all_time_steps)} 个，"
        f"frame_step={frame_step}，导出帧数={len(sampled)}。"
    )
    return sampled


def build_heat_matrices(
    spatial: pd.DataFrame,
) -> tuple[dict[float, np.ndarray], list[float], list[float]]:
    """构建各时间步的数据中心与机房负载热力矩阵。"""
    dc_ids = sorted(spatial["datacenter_id"].dropna().unique().tolist())
    room_ids = sorted(spatial["server_room_id"].dropna().unique().tolist())
    matrices: dict[float, np.ndarray] = {}
    for time_step, frame in spatial.groupby("time_step"):
        pivot = frame.pivot_table(
            index="datacenter_id",
            columns="server_room_id",
            values="load_rate",
            aggfunc="mean",
            fill_value=0,
        )
        pivot = pivot.reindex(index=dc_ids, columns=room_ids, fill_value=0)
        matrices[float(time_step)] = pivot.to_numpy(dtype=float)
    return matrices, dc_ids, room_ids


def build_metrics(
    spatial: pd.DataFrame,
    cluster: pd.DataFrame,
    server_step: pd.DataFrame,
    chip: pd.DataFrame,
) -> pd.DataFrame:
    """计算动画所需的小时级负载、任务和功率指标。"""
    totals = (
        spatial.groupby("time_step", as_index=False)
        .agg(
            total_cpu_usage=("cpu_usage", "sum"),
            total_task_count=("task_count", "sum"),
            avg_cpu_load_rate=("load_rate", "mean"),
            max_cpu_load_rate=("load_rate", "max"),
        )
        .sort_values("time_step")
    )
    metrics = totals
    for extra in [cluster, server_step, chip]:
        if extra is not None and not extra.empty and "time_step" in extra.columns:
            metrics = metrics.merge(extra, on="time_step", how="left")
    return metrics.sort_values("time_step").reset_index(drop=True)


def make_colormap() -> LinearSegmentedColormap:
    """创建负载热力图使用的颜色映射。"""
    return LinearSegmentedColormap.from_list(
        "academic_blues",
        ["#f8fbff", "#dceeff", "#9ecae1", "#3a83c1", "#0b3a67"],
    )


def create_animation(
    spatial: pd.DataFrame,
    metrics: pd.DataFrame,
    sampled_times: list[float],
    fps: int,
) -> animation.FuncAnimation:
    """绘图部分：热力图 + 右侧当前指标 + 时间趋势。"""
    matrices, dc_ids, room_ids = build_heat_matrices(spatial)
    metrics_by_time = metrics.set_index("time_step")
    all_times = metrics["time_step"].to_numpy(dtype=float)
    avg_load_rate = metrics["avg_cpu_load_rate"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(13.5, 7.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.4, 1.35],
        height_ratios=[3.0, 1.15],
        wspace=0.28,
        hspace=0.3,
    )
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    ax_trend = fig.add_subplot(grid[1, 1])

    fig.suptitle(TITLE, fontsize=18, fontweight="bold", color="#0b3a67", y=0.98)

    first_time = float(sampled_times[0])
    first_matrix = matrices.get(first_time)
    if first_matrix is None:
        first_matrix = next(iter(matrices.values()))

    image = ax_heat.imshow(
        first_matrix,
        cmap=make_colormap(),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("CPU 负载率", fontsize=10, color="#17324d")
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax_heat.set_title(HEATMAP_TITLE, fontsize=13, fontweight="bold", color="#17324d", pad=12)
    ax_heat.set_xlabel("机房编号", fontsize=11)
    ax_heat.set_ylabel("数据中心编号", fontsize=11)
    ax_heat.set_xticks(np.arange(len(room_ids)))
    ax_heat.set_xticklabels([f"机房{int(room)}" for room in room_ids], rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(dc_ids)))
    ax_heat.set_yticklabels([f"数据中心{int(dc)}" for dc in dc_ids])
    ax_heat.set_xticks(np.arange(-0.5, len(room_ids), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(dc_ids), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.8)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    ax_info.axis("off")
    info_text = ax_info.text(
        0.05,
        0.95,
        "",
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="#17324d",
        linespacing=1.6,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#f3f8ff",
            "edgecolor": "#b9d7f0",
            "linewidth": 1.2,
        },
    )

    ax_trend.plot(all_times, avg_load_rate, color="#1f6fb2", linewidth=1.8)
    current_line = ax_trend.axvline(first_time, color="#0b3a67", linewidth=1.4)
    current_marker = ax_trend.plot(
        [first_time],
        [metrics_by_time.loc[first_time, "avg_cpu_load_rate"]],
        marker="o",
        markersize=5,
        color="#0b3a67",
    )[0]
    ax_trend.set_title("平均 CPU 负载率趋势", fontsize=10, color="#17324d")
    ax_trend.set_xlabel("时间步", fontsize=9)
    ax_trend.set_ylabel("负载率", fontsize=9)
    ax_trend.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_trend.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax_trend.tick_params(labelsize=8)
    ax_trend.set_ylim(0, 1)

    def metric_at(time_step: float, column: str) -> object:
        """读取指定帧对应的指标值，缺失时返回默认值。"""
        if column not in metrics_by_time.columns or time_step not in metrics_by_time.index:
            return np.nan
        value = metrics_by_time.loc[time_step, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return value

    def update(time_step: float):
        """根据当前帧刷新动画中的图层和指标展示。"""
        matrix = matrices.get(float(time_step))
        if matrix is not None:
            image.set_data(matrix)

        avg_load_value = metric_at(time_step, "avg_cpu_load_rate")
        max_load_value = metric_at(time_step, "max_cpu_load_rate")
        total_task_value = metric_at(time_step, "total_task_count")
        cluster_cap = metric_at(time_step, "cluster_power_cap")
        electricity_price = metric_at(time_step, "electricity_price")

        info_lines = [
            f"当前时间步：{format_time_step(time_step)}",
            f"平均 CPU 负载率：{format_percent(avg_load_value)}",
            f"最大 CPU 负载率：{format_percent(max_load_value)}",
            f"任务处理数量：{format_value(total_task_value)}",
            f"归一化功率上限：{format_value(cluster_cap, 3)}",
            f"电价：{format_value(electricity_price, 2, ' 美元/MWh')}",
        ]

        info_text.set_text("\n".join(info_lines))
        current_line.set_xdata([time_step, time_step])
        current_marker.set_data([time_step], [float(metric_at(time_step, "avg_cpu_load_rate"))])
        return image, info_text, current_line, current_marker

    interval_ms = int(1000 / max(fps, 1))
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=[float(time) for time in sampled_times],
        interval=interval_ms,
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )
    anim._codex_figure = fig  # 保留引用，导出后关闭。
    return anim


def save_animation(
    anim: animation.FuncAnimation,
    frame_count: int,
    output_paths: OutputPaths,
    *,
    fps: int,
    dpi: int,
    skip_mp4: bool,
) -> None:
    """动图导出部分：GIF 必导出，MP4 视 ffmpeg 可用性导出。"""
    output_paths.gif.parent.mkdir(parents=True, exist_ok=True)

    gif_writer = animation.PillowWriter(fps=fps)
    with tqdm(total=frame_count, desc="导出 GIF") as progress:
        def gif_progress(current_frame: int, total_frames: int) -> None:
            """接收 GIF 导出进度并更新进度条。"""
            if total_frames and progress.total != total_frames:
                progress.total = total_frames
            progress.update(max(0, current_frame + 1 - progress.n))

        anim.save(
            output_paths.gif,
            writer=gif_writer,
            dpi=dpi,
            progress_callback=gif_progress,
        )
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

            anim.save(
                output_paths.mp4,
                writer=mp4_writer,
                dpi=dpi,
                progress_callback=mp4_progress,
            )
        print(f"[完成] MP4 已生成：{output_paths.mp4}")
    except Exception as exc:  # ffmpeg 配置异常时不终止 GIF 结果。
        print(f"MP4 导出失败，可能是未安装 FFmpeg；GIF 已正常生成。错误信息：{exc}")


# 答辩说明：
# 该动图基于服务器级调度数据，按照时间步统计不同数据中心和机房的 CPU 负载率，
# 并结合集群级功率上限与电价信息，展示功率约束和价格信号影响下数据中心负载
# 在内部层级空间中的动态分布过程。图中颜色越深表示对应数据中心—机房组合的
# CPU 负载率越高，右侧信息框展示当前时刻的平均负载率、最大负载率、任务处理
# 数量、归一化功率上限和电价。


def main() -> int:
    """作为脚本入口协调参数解析、数据处理和结果导出。"""
    args = parse_args()
    configure_chinese_font()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = OutputPaths(
        gif=output_dir / "spatiotemporal_load_rate_animation.gif",
        mp4=output_dir / "spatiotemporal_load_rate_animation.mp4",
    )

    print(f"[路径] 数据目录：{data_dir}")
    print(f"[路径] 输出目录：{output_dir}")

    server_path = find_excel_file(data_dir, SERVER_FILE_NAME, ["服务器"])
    cluster_path = find_excel_file(data_dir, CLUSTER_FILE_NAME, ["集群"])
    chip_path = find_excel_file(data_dir, CHIP_FILE_NAME, ["芯片"])

    # 自动检查 sheet 名称、字段名称和数据结构。
    for path in [server_path, cluster_path, chip_path]:
        inspect_workbook(path)

    # 数据读取、字段识别与聚合。
    spatial = load_spatial_data(server_path)
    all_time_steps = sorted(float(value) for value in spatial["time_step"].dropna().unique())
    cluster = load_cluster_metadata(cluster_path)
    server_step = load_server_step_metadata(server_path)
    chip = load_chip_metadata(chip_path)
    metrics = build_metrics(spatial, cluster, server_step, chip)
    sampled_times = choose_sampled_times(all_time_steps, args.frame_step, args.max_frames)

    # 绘图与导出。
    anim = create_animation(spatial, metrics, sampled_times, args.fps)
    try:
        save_animation(
            anim,
            len(sampled_times),
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
