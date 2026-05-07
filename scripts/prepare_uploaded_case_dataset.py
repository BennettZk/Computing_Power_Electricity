from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shutil
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.task import load_tasks_csv
from utils.io_utils import load_all_configs


DEFAULT_CLUSTER_XLSX = PROJECT_ROOT / "data" / "raw" / "数据中心集群级别的调度数据.xlsx"
DEFAULT_SERVER_XLSX = PROJECT_ROOT / "data" / "raw" / "服务器级别的调度数据.xlsx"
DEFAULT_CHIP_XLSX = PROJECT_ROOT / "data" / "raw" / "芯片级别的调度数据.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "real_case"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "uploaded_case_cleaning_report.txt"
SERVER_SHEET_NAME = "0821_schedule_fcfs_random_with_"
CHIP_SHEET_NAME = "output3_chip"
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


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cell_col_index(cell_ref: str | None, fallback: int) -> int:
    if not cell_ref:
        return fallback
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    if not letters:
        return fallback
    index = 0
    for char in letters.upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _parse_scalar(value: str | None):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


class SimpleXlsxWorkbook:
    """Small dependency-free XLSX reader for value-only worksheets."""

    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    pkg_rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists():
            raise FileNotFoundError(f"XLSX file not found: {path}")
        self._zip = ZipFile(path)
        self._shared_strings = self._load_shared_strings()
        self._sheet_paths = self._load_sheet_paths()

    @property
    def sheet_names(self) -> list[str]:
        return list(self._sheet_paths.keys())

    def close(self) -> None:
        self._zip.close()

    def _load_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self._zip.namelist():
            return []
        root = ET.fromstring(self._zip.read("xl/sharedStrings.xml"))
        strings: list[str] = []
        for item in root.findall(f"{self.main_ns}si"):
            texts = [node.text or "" for node in item.findall(f".//{self.main_ns}t")]
            strings.append("".join(texts))
        return strings

    def _load_sheet_paths(self) -> dict[str, str]:
        workbook_root = ET.fromstring(self._zip.read("xl/workbook.xml"))
        rel_root = ET.fromstring(self._zip.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall(f"{self.pkg_rel_ns}Relationship")
        }

        sheet_paths: dict[str, str] = {}
        for sheet in workbook_root.findall(f".//{self.main_ns}sheet"):
            name = sheet.attrib.get("name", "")
            rel_id = sheet.attrib.get(f"{self.rel_ns}id")
            target = rel_targets.get(rel_id or "")
            if not name or not target:
                continue
            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            elif target.startswith("xl/"):
                sheet_path = target
            else:
                sheet_path = f"xl/{target}"
            sheet_paths[name] = sheet_path
        return sheet_paths

    def read_sheet(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self._sheet_paths:
            raise ValueError(f"Sheet {sheet_name!r} not found in {self.path}. Available sheets: {self.sheet_names}")
        root = ET.fromstring(self._zip.read(self._sheet_paths[sheet_name]))
        rows: list[list[object]] = []
        for row_node in root.findall(f".//{self.main_ns}sheetData/{self.main_ns}row"):
            row_values: list[object] = []
            for fallback_idx, cell in enumerate(row_node.findall(f"{self.main_ns}c")):
                col_idx = _cell_col_index(cell.attrib.get("r"), fallback_idx)
                while len(row_values) <= col_idx:
                    row_values.append(None)
                row_values[col_idx] = self._cell_value(cell)
            if any(value is not None for value in row_values):
                rows.append(row_values)

        if not rows:
            return pd.DataFrame()
        width = max(len(row) for row in rows)
        rows = [row + [None] * (width - len(row)) for row in rows]
        raw_headers = rows[0]
        headers = self._dedupe_headers(raw_headers)
        return pd.DataFrame(rows[1:], columns=headers)

    def _cell_value(self, cell):
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            texts = [node.text or "" for node in cell.findall(f".//{self.main_ns}t")]
            return "".join(texts) if texts else None

        value_node = cell.find(f"{self.main_ns}v")
        raw_value = value_node.text if value_node is not None else None
        if cell_type == "s":
            try:
                return self._shared_strings[int(raw_value or 0)]
            except (IndexError, ValueError):
                return raw_value
        if cell_type == "b":
            return str(raw_value).strip() == "1"
        if cell_type in {"str", "e"}:
            return raw_value
        return _parse_scalar(raw_value)

    @staticmethod
    def _dedupe_headers(raw_headers: list[object]) -> list[str]:
        headers: list[str] = []
        seen: dict[str, int] = {}
        for idx, header in enumerate(raw_headers):
            name = str(header).strip() if header is not None else ""
            if not name:
                name = f"unnamed_{idx + 1}"
            count = seen.get(name, 0)
            seen[name] = count + 1
            headers.append(name if count == 0 else f"{name}_{count + 1}")
        return headers


def _load_workbook(path: Path) -> SimpleXlsxWorkbook:
    try:
        return SimpleXlsxWorkbook(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read XLSX file {path}: {exc}") from exc


def _normalize_column(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    normalized = {_normalize_column(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Missing required column. Expected one of {candidates}; got {list(df.columns)}")
    return None


def _find_sheet(sheet_names: list[str], preferred: str, description: str) -> str:
    if preferred in sheet_names:
        return preferred
    normalized_preferred = _normalize_column(preferred)
    for sheet_name in sheet_names:
        normalized_name = _normalize_column(sheet_name)
        if normalized_preferred in normalized_name or normalized_name in normalized_preferred:
            return sheet_name
    raise ValueError(f"Missing {description} sheet {preferred!r}. Available sheets: {sheet_names}")


def _numeric_series(
    df: pd.DataFrame,
    col: str | None,
    default,
    fill_counter: dict[str, int],
    name: str,
    clip_lower: float | None = None,
) -> pd.Series:
    if isinstance(default, pd.Series):
        default_series = default.reset_index(drop=True).astype(float)
    else:
        default_series = pd.Series([float(default)] * len(df), dtype=float)

    if col is None:
        fill_counter[name] = fill_counter.get(name, 0) + len(df)
        result = default_series.copy()
    else:
        raw = pd.to_numeric(df[col], errors="coerce")
        missing = int(raw.isna().sum())
        if missing:
            fill_counter[name] = fill_counter.get(name, 0) + missing
        result = raw.fillna(default_series).astype(float)

    if clip_lower is not None:
        result = result.clip(lower=clip_lower)
    return result.reset_index(drop=True)


def _text_series(df: pd.DataFrame, col: str, default_prefix: str, fill_counter: dict[str, int], name: str) -> pd.Series:
    values = df[col].astype("string").reset_index(drop=True)
    missing = int(values.isna().sum() + (values.fillna("").str.strip() == "").sum())
    if missing:
        fill_counter[name] = fill_counter.get(name, 0) + missing
    fallback = pd.Series([f"{default_prefix}-{idx:06d}" for idx in range(len(df))], dtype="string")
    values = values.where(values.fillna("").str.strip() != "", fallback)
    return values.fillna(fallback).astype(str)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_server_data(
    workbook: SimpleXlsxWorkbook,
    output_dir: Path,
    seed: int,
    fill_counter: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sheet_name = _find_sheet(workbook.sheet_names, SERVER_SHEET_NAME, "server scheduling")
    df = workbook.read_sheet(sheet_name)
    if df.empty:
        raise ValueError(f"Server scheduling sheet {sheet_name!r} is empty.")

    job_col = _find_column(df, ["Job", "job", "task_id"], required=True)
    arrival_col = _find_column(df, ["Arrival time", "arrival_time"], required=True)
    scheduled_col = _find_column(df, ["Scheduled start time", "scheduled_start_time"])
    execution_col = _find_column(df, ["Execution time", "execution_time"])
    cpu_col = _find_column(df, ["CPU", "cpu", "cpu_demand"])
    end_col = _find_column(df, ["end_step", "End step", "end step"])

    task_id = _text_series(df, job_col, "uploaded-job", fill_counter, "task_id")
    arrival_step = _numeric_series(df, arrival_col, 0.0, fill_counter, "arrival_time", clip_lower=0.0)
    scheduled_step = _numeric_series(df, scheduled_col, arrival_step, fill_counter, "scheduled_start_time", clip_lower=0.0)
    execution_steps = _numeric_series(df, execution_col, 12.0, fill_counter, "execution_time", clip_lower=0.0)
    cpu_demand = _numeric_series(df, cpu_col, 1.0, fill_counter, "cpu_demand", clip_lower=0.0)

    arrival_hour = np.floor(arrival_step / 12.0).astype(int).clip(0, 23)
    fallback_deadline = arrival_hour + np.ceil(execution_steps / 12.0).astype(int) + 1
    if end_col is not None:
        end_step = pd.to_numeric(df[end_col], errors="coerce").reset_index(drop=True)
        missing_end = int(end_step.isna().sum())
        if missing_end:
            fill_counter["end_step"] = fill_counter.get("end_step", 0) + missing_end
        deadline_raw = np.floor(end_step.fillna(fallback_deadline * 12.0) / 12.0).astype(int)
    else:
        fill_counter["end_step"] = fill_counter.get("end_step", 0) + len(df)
        end_step = pd.Series([np.nan] * len(df), dtype=float)
        deadline_raw = fallback_deadline.astype(int)

    deadline_before_fix = pd.Series(deadline_raw, dtype=int)
    deadline_anomaly_before = int((deadline_before_fix < arrival_hour).sum())
    deadline = np.maximum(deadline_before_fix, arrival_hour).clip(0, 23).astype(int)

    waiting_time = scheduled_step - arrival_step
    task_type = np.where(waiting_time <= 1.0, "delay_sensitive", "delay_tolerant")
    priority = np.where(task_type == "delay_sensitive", 3, 2)
    rng = np.random.default_rng(seed)
    migratable = (task_type == "delay_tolerant") & (rng.random(len(df)) < 0.30)
    memory_demand = np.maximum(4.0, cpu_demand.to_numpy(dtype=float) * 0.5)
    migration_delay = np.where(migratable, 0.05, 0.0)

    tasks_df = pd.DataFrame(
        {
            "task_id": task_id,
            "arrival_time": arrival_hour.astype(int),
            "task_type": task_type,
            "cpu_demand": cpu_demand.astype(float),
            "gpu_demand": 0.0,
            "memory_demand": memory_demand.astype(float),
            "bandwidth_demand": 0.5,
            "token_amount": 0.0,
            "deadline": deadline.astype(int),
            "priority": priority.astype(int),
            "migratable": migratable.astype(bool),
            "migration_cost_weight": 1.0,
            "migration_delay_penalty": migration_delay.astype(float),
        }
    )

    raw_mapped = pd.DataFrame(
        {
            "task_id": task_id,
            "arrival_step": arrival_step,
            "scheduled_start_step": scheduled_step,
            "execution_time_steps": execution_steps,
            "end_step": end_step,
            "waiting_time_steps": waiting_time,
            "arrival_time": arrival_hour.astype(int),
            "deadline": deadline.astype(int),
            "task_type": task_type,
            "cpu_demand": cpu_demand.astype(float),
            "gpu_demand": 0.0,
            "memory_demand": memory_demand.astype(float),
            "bandwidth_demand": 0.5,
            "token_amount": 0.0,
            "priority": priority.astype(int),
            "migratable": migratable.astype(bool),
            "migration_cost_weight": 1.0,
            "migration_delay_penalty": migration_delay.astype(float),
        }
    )
    for col in ["datacenter_id", "server_room_id", "rack_id", "server_id", "global_server_id"]:
        original_col = _find_column(df, [col])
        if original_col is not None:
            raw_mapped[col] = df[original_col].reset_index(drop=True)

    _write_csv(tasks_df, output_dir / "server_tasks_24h.csv")
    _write_csv(raw_mapped, output_dir / "server_tasks_5min_raw_mapped.csv")

    load_tasks_csv(output_dir / "server_tasks_24h.csv")
    meta = {
        "sheet_name": sheet_name,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "deadline_anomaly_before": deadline_anomaly_before,
        "deadline_anomaly_after": int((tasks_df["deadline"] < tasks_df["arrival_time"]).sum()),
    }
    return tasks_df, raw_mapped, meta


def clean_cluster_data(
    workbook: SimpleXlsxWorkbook,
    output_dir: Path,
    server_tasks_df: pd.DataFrame,
    fill_counter: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not workbook.sheet_names:
        raise ValueError(f"Cluster workbook {workbook.path} has no sheets.")
    sheet_name = workbook.sheet_names[0]
    df = workbook.read_sheet(sheet_name)
    if df.empty:
        raise ValueError(f"Cluster sheet {sheet_name!r} is empty.")

    time_col = _find_column(df, ["时间步(5分钟)", "时间步（5分钟）", "time_step", "Time step", "step"])
    power_col = _find_column(df, ["功率上限（标幺值）", "功率上限(标幺值)", "power_cap_pu"])
    price_col = _find_column(df, ["电价（美元/兆瓦时）", "电价(美元/兆瓦时)", "price"])
    reward_col = _find_column(df, ["奖励值", "reward"])

    row_index = pd.Series(np.arange(len(df)), dtype=float)
    time_step = _numeric_series(df, time_col, row_index, fill_counter, "cluster_time_step", clip_lower=0.0)
    power_cap = _numeric_series(df, power_col, 1.0, fill_counter, "power_cap_pu", clip_lower=0.0)
    price = _numeric_series(df, price_col, 0.0, fill_counter, "price", clip_lower=0.0)
    reward = _numeric_series(df, reward_col, 0.0, fill_counter, "reward")

    cluster_5min = pd.DataFrame(
        {
            "time_step": time_step,
            "power_cap_pu": power_cap,
            "price": price,
            "reward": reward,
        }
    )
    _write_csv(cluster_5min, output_dir / "cluster_power_price_5min.csv")

    cluster_5min["hour"] = np.floor(cluster_5min["time_step"] / 12.0).astype(int).clip(0, 23)
    grouped = cluster_5min.groupby("hour")[["price", "power_cap_pu"]].mean().reindex(range(24))
    grouped["price"] = grouped["price"].fillna(grouped["price"].mean() if not grouped["price"].dropna().empty else 0.0)
    grouped["power_cap_pu"] = grouped["power_cap_pu"].fillna(
        grouped["power_cap_pu"].mean() if not grouped["power_cap_pu"].dropna().empty else 1.0
    )
    arrival_rate = server_tasks_df.groupby("arrival_time").size().reindex(range(24), fill_value=0).astype(int)
    hourly = pd.DataFrame(
        {
            "hour": range(24),
            "arrival_rate": arrival_rate.to_numpy(dtype=int),
            "price": grouped["price"].to_numpy(dtype=float),
            "carbon_factor": DEFAULT_CARBON,
            "power_limit": grouped["power_cap_pu"].to_numpy(dtype=float),
        }
    )
    _write_csv(hourly, output_dir / "hourly_input_24h.csv")
    meta = {
        "sheet_name": sheet_name,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
    }
    return cluster_5min, hourly, meta


def clean_chip_data(
    workbook: SimpleXlsxWorkbook,
    output_dir: Path,
    fill_counter: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    sheet_name = _find_sheet(workbook.sheet_names, CHIP_SHEET_NAME, "chip DVFS")
    df = workbook.read_sheet(sheet_name)
    if df.empty:
        raise ValueError(f"Chip sheet {sheet_name!r} is empty.")

    actual_col = _find_column(df, ["实际功率(瓦)", "实际功率（瓦）", "actual_power_w"])
    reported_col = _find_column(df, ["报量功率（瓦）", "报量功率(瓦)", "reported_power_w"])
    cap_col = _find_column(df, ["功率上限（瓦）", "功率上限(瓦)", "power_cap_w"])
    freq_col = _find_column(df, ["频率(10^9Hz)", "频率（10^9Hz）", "frequency_ghz"])

    chip = pd.DataFrame(
        {
            "actual_power_w": _numeric_series(df, actual_col, 0.0, fill_counter, "actual_power_w", clip_lower=0.0),
            "reported_power_w": _numeric_series(df, reported_col, 0.0, fill_counter, "reported_power_w", clip_lower=0.0),
            "power_cap_w": _numeric_series(df, cap_col, 0.0, fill_counter, "power_cap_w", clip_lower=0.0),
            "frequency_ghz": _numeric_series(df, freq_col, 0.0, fill_counter, "frequency_ghz", clip_lower=0.0),
        }
    )
    _write_csv(chip, output_dir / "chip_dvfs.csv")
    meta = {
        "sheet_name": sheet_name,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
    }
    return chip, meta


def replace_main_inputs(output_dir: Path, report_lines: list[str]) -> None:
    tasks_source = output_dir / "server_tasks_24h.csv"
    hourly_source = output_dir / "hourly_input_24h.csv"
    tasks_target = PROJECT_ROOT / "data" / "synthetic" / "tasks.csv"
    hourly_target = PROJECT_ROOT / "data" / "hourly_input.csv"
    tasks_backup = PROJECT_ROOT / "data" / "synthetic" / "tasks_backup_before_real_case.csv"
    hourly_backup = PROJECT_ROOT / "data" / "hourly_input_backup_before_real_case.csv"
    processed_cache = PROJECT_ROOT / "data" / "processed" / "hourly_profile.csv"

    if tasks_target.exists():
        shutil.copy2(tasks_target, tasks_backup)
    if hourly_target.exists():
        shutil.copy2(hourly_target, hourly_backup)
    shutil.copy2(tasks_source, tasks_target)
    shutil.copy2(hourly_source, hourly_target)
    if processed_cache.exists():
        processed_cache.unlink()

    report_lines.extend(
        [
            "",
            "主输入替换:",
            f"- 已备份任务输入到: {tasks_backup}",
            f"- 已备份小时输入到: {hourly_backup}",
            f"- 已替换任务输入: {tasks_target}",
            f"- 已替换小时输入: {hourly_target}",
            f"- 已删除小时缓存: {processed_cache if not processed_cache.exists() else '删除失败'}",
        ]
    )


def _format_distribution(series: pd.Series) -> str:
    return series.to_string()


def build_report(
    workbooks: dict[str, SimpleXlsxWorkbook],
    server_meta: dict[str, object],
    cluster_meta: dict[str, object],
    chip_meta: dict[str, object],
    server_tasks_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    chip_df: pd.DataFrame,
    fill_counter: dict[str, int],
) -> list[str]:
    arrival_distribution = server_tasks_df.groupby("arrival_time").size().reindex(range(24), fill_value=0)
    task_type_distribution = server_tasks_df["task_type"].value_counts()
    hourly_overview = hourly_df[["hour", "arrival_rate", "price", "power_limit"]].to_string(index=False)

    lines = ["上传 XLSX 数据清洗报告", ""]
    lines.append("Sheet 名列表:")
    for name, workbook in workbooks.items():
        lines.append(f"- {name}: {workbook.sheet_names}")

    lines.extend(
        [
            "",
            "关键 Sheet 概览:",
            f"- 服务器任务 Sheet: {server_meta['sheet_name']}, 行数={server_meta['rows']}, 列数={server_meta['cols']}",
            f"  字段名: {server_meta['columns']}",
            f"- 集群场景 Sheet: {cluster_meta['sheet_name']}, 行数={cluster_meta['rows']}, 列数={cluster_meta['cols']}",
            f"  字段名: {cluster_meta['columns']}",
            f"- 芯片 DVFS Sheet: {chip_meta['sheet_name']}, 行数={chip_meta['rows']}, 列数={chip_meta['cols']}",
            f"  字段名: {chip_meta['columns']}",
            "",
            f"服务器任务总数: {len(server_tasks_df)}",
            "",
            "arrival_time 的 0-23 小时分布:",
            _format_distribution(arrival_distribution),
            "",
            "task_type 分布:",
            task_type_distribution.to_string(),
            "",
            "cpu_demand 统计:",
            f"- mean={server_tasks_df['cpu_demand'].mean():.6f}",
            f"- min={server_tasks_df['cpu_demand'].min():.6f}",
            f"- max={server_tasks_df['cpu_demand'].max():.6f}",
            "",
            "deadline 异常检查:",
            f"- 修正前 deadline < arrival_time 数量: {server_meta['deadline_anomaly_before']}",
            f"- 清洗后 deadline < arrival_time 数量: {server_meta['deadline_anomaly_after']}",
            "",
            "hourly_input_24h.csv 概览:",
            hourly_overview,
            "",
            "chip_dvfs.csv 概览:",
            f"- 行数: {len(chip_df)}",
            f"- actual_power_w 范围: {chip_df['actual_power_w'].min():.6f} - {chip_df['actual_power_w'].max():.6f}",
            f"- frequency_ghz 范围: {chip_df['frequency_ghz'].min():.6f} - {chip_df['frequency_ghz'].max():.6f}",
            "",
            "缺失值填补数量:",
        ]
    )
    if fill_counter:
        lines.extend([f"- {key}: {value}" for key, value in sorted(fill_counter.items())])
    else:
        lines.append("- 无")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗上传的集群/服务器/芯片级 XLSX 数据为项目标准 CSV。")
    parser.add_argument("--cluster-xlsx", default=str(DEFAULT_CLUSTER_XLSX), help="数据中心集群级别 xlsx 路径。")
    parser.add_argument("--server-xlsx", default=str(DEFAULT_SERVER_XLSX), help="服务器级别 xlsx 路径。")
    parser.add_argument("--chip-xlsx", default=str(DEFAULT_CHIP_XLSX), help="芯片级别 xlsx 路径。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="清洗后 CSV 输出目录。")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH), help="清洗报告输出路径。")
    parser.add_argument("--seed", type=int, help="随机种子；默认读取 config/experiment.yaml。")
    parser.add_argument("--replace-main-inputs", action="store_true", help="备份并替换 data/synthetic/tasks.csv 和 data/hourly_input.csv。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_all_configs(PROJECT_ROOT / "config")
    seed = int(args.seed if args.seed is not None else configs["experiment"]["seed"])
    output_dir = _project_path(args.output_dir)
    report_path = _project_path(args.report_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    workbooks = {
        "集群级数据": _load_workbook(_project_path(args.cluster_xlsx)),
        "服务器级数据": _load_workbook(_project_path(args.server_xlsx)),
        "芯片级数据": _load_workbook(_project_path(args.chip_xlsx)),
    }
    try:
        fill_counter: dict[str, int] = {}
        server_tasks_df, _, server_meta = clean_server_data(workbooks["服务器级数据"], output_dir, seed, fill_counter)
        _, hourly_df, cluster_meta = clean_cluster_data(workbooks["集群级数据"], output_dir, server_tasks_df, fill_counter)
        chip_df, chip_meta = clean_chip_data(workbooks["芯片级数据"], output_dir, fill_counter)

        report_lines = build_report(
            workbooks=workbooks,
            server_meta=server_meta,
            cluster_meta=cluster_meta,
            chip_meta=chip_meta,
            server_tasks_df=server_tasks_df,
            hourly_df=hourly_df,
            chip_df=chip_df,
            fill_counter=fill_counter,
        )
        if args.replace_main_inputs:
            replace_main_inputs(output_dir, report_lines)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
    finally:
        for workbook in workbooks.values():
            workbook.close()

    print(f"Server tasks written to: {output_dir / 'server_tasks_24h.csv'}")
    print(f"Hourly input written to: {output_dir / 'hourly_input_24h.csv'}")
    print(f"Chip DVFS data written to: {output_dir / 'chip_dvfs.csv'}")
    print(f"Report written to: {report_path}")
    if args.replace_main_inputs:
        print("Main inputs replaced after backup. Run main.py only after confirming the cleaned files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
