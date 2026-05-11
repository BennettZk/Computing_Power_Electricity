from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import save_csv, write_text


DEFAULT_CLUSTER_XLSX = PROJECT_ROOT / "data" / "raw" / "数据中心集群级别的调度数据.xlsx"
DEFAULT_SERVER_XLSX = PROJECT_ROOT / "data" / "raw" / "服务器级别的调度数据.xlsx"
DEFAULT_CHIP_XLSX = PROJECT_ROOT / "data" / "raw" / "芯片级别的调度数据.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "real_case"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "uploaded_case_cleaning_report.txt"
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
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


class SimpleXlsxWorkbook:
    """Dependency-free XLSX reader for value-only worksheets."""

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
        headers = self._dedupe_headers(rows[0])
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


def _read_workbook(path: Path) -> SimpleXlsxWorkbook:
    try:
        return SimpleXlsxWorkbook(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read XLSX file {path}: {exc}") from exc


def _normalize_column(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _find_column(df: pd.DataFrame, candidates: list[str], warnings: list[str], required: bool = False) -> str | None:
    normalized = {_normalize_column(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    message = f"Missing column, expected one of {candidates}; available columns: {list(df.columns)}"
    if required:
        raise ValueError(message)
    warnings.append(message)
    return None


def _numeric_series(df: pd.DataFrame, column: str | None, default: float, fill_counts: dict[str, int], name: str) -> pd.Series:
    if column is None:
        fill_counts[name] = len(df)
        return pd.Series(default, index=df.index, dtype="float64")
    raw = pd.to_numeric(df[column], errors="coerce")
    fill_counts[name] = int(raw.isna().sum())
    return raw.fillna(default).astype("float64")


def _text_series(df: pd.DataFrame, column: str | None, default: str, fill_counts: dict[str, int], name: str) -> pd.Series:
    if column is None:
        fill_counts[name] = len(df)
        return pd.Series(default, index=df.index, dtype="object")
    raw = df[column].astype("object")
    fill_counts[name] = int(raw.isna().sum())
    return raw.fillna(default).astype(str)


def _select_sheet(workbook: SimpleXlsxWorkbook, preferred: str | None, warnings: list[str]) -> tuple[str, pd.DataFrame]:
    if preferred and preferred in workbook.sheet_names:
        sheet_name = preferred
    else:
        sheet_name = workbook.sheet_names[0]
        if preferred:
            warnings.append(f"Sheet {preferred!r} not found in {workbook.path.name}; using first sheet {sheet_name!r}.")
    return sheet_name, workbook.read_sheet(sheet_name)


def _map_steps_to_hours(step_series: pd.Series, min_step: float, max_step: float, mode: str) -> pd.Series:
    step_series = pd.Series(step_series, dtype="float64").fillna(min_step)
    if mode == "raw_step":
        return np.floor(step_series / 12).astype(int).clip(0, 23)
    span = max(max_step - min_step, 1e-9)
    return np.floor((step_series - min_step) / span * 23).astype(int).clip(0, 23)


def _resolve_arrival_time_mode(arrival_step: pd.Series, requested_mode: str) -> tuple[str, dict]:
    arrival_step = pd.Series(arrival_step, dtype="float64").fillna(0.0)
    min_step = float(arrival_step.min()) if len(arrival_step) else 0.0
    max_step = float(arrival_step.max()) if len(arrival_step) else 0.0
    raw_hours = np.floor(arrival_step / 12).astype(int).clip(0, 23)
    raw_counts = raw_hours.value_counts().reindex(range(24), fill_value=0)
    nonzero_hours = int((raw_counts > 0).sum())
    late_hour_share = float(raw_counts.loc[8:23].sum() / max(len(arrival_step), 1))
    concentrated = nonzero_hours <= 10 or late_hour_share < 0.05
    should_rescale = max_step < 288 or concentrated
    if requested_mode == "auto":
        resolved = "rescale_24h" if should_rescale else "raw_step"
    else:
        resolved = requested_mode
    diagnostics = {
        "arrival_step_min": min_step,
        "arrival_step_max": max_step,
        "raw_nonzero_hours": nonzero_hours,
        "raw_late_hour_share": late_hour_share,
        "auto_should_rescale": bool(should_rescale),
        "arrival_time_mode_requested": requested_mode,
        "arrival_time_mode_resolved": resolved,
    }
    return resolved, diagnostics


def clean_server_data(server_xlsx: Path, output_dir: Path, seed: int, arrival_time_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    warnings: list[str] = []
    fill_counts: dict[str, int] = {}
    workbook = _read_workbook(server_xlsx)
    try:
        sheet_name, df = _select_sheet(workbook, SERVER_SHEET_NAME, warnings)
        sheet_names = workbook.sheet_names
    finally:
        workbook.close()

    job_col = _find_column(df, ["Job", "job", "task_id"], warnings)
    arrival_col = _find_column(df, ["Arrival time", "arrival_time", "arrival step"], warnings, required=True)
    start_col = _find_column(df, ["Scheduled start time", "scheduled_start_time", "start time"], warnings)
    exec_col = _find_column(df, ["Execution time", "execution_time", "duration"], warnings)
    cpu_col = _find_column(df, ["CPU", "cpu", "cpu_demand"], warnings)
    end_col = _find_column(df, ["end_step", "End step", "finish time"], warnings)
    dc_col = _find_column(df, ["datacenter_id", "data_center_id"], warnings)
    room_col = _find_column(df, ["server_room_id", "room_id", "机房"], warnings)
    rack_col = _find_column(df, ["rack_id", "机架"], warnings)
    server_col = _find_column(df, ["server_id", "服务器"], warnings)
    global_server_col = _find_column(df, ["global_server_id", "global server id"], warnings)

    rng = np.random.default_rng(seed)
    arrival_step = _numeric_series(df, arrival_col, 0.0, fill_counts, "arrival_step")
    scheduled_start_step = _numeric_series(df, start_col, arrival_step.mean() if len(arrival_step) else 0.0, fill_counts, "scheduled_start_step")
    scheduled_start_step = scheduled_start_step.where(scheduled_start_step >= arrival_step, arrival_step)
    execution_time_steps = _numeric_series(df, exec_col, 1.0, fill_counts, "execution_time_steps").clip(lower=1.0)
    end_step_default = scheduled_start_step + execution_time_steps
    end_step = _numeric_series(df, end_col, 0.0, fill_counts, "end_step")
    end_step = end_step.where(end_step > 0, end_step_default)
    cpu_demand = _numeric_series(df, cpu_col, 1.0, fill_counts, "cpu_demand").clip(lower=0.1)

    resolved_mode, time_mapping_meta = _resolve_arrival_time_mode(arrival_step, arrival_time_mode)
    min_arrival_step = float(time_mapping_meta["arrival_step_min"])
    max_arrival_step = float(time_mapping_meta["arrival_step_max"])
    arrival_hour = _map_steps_to_hours(arrival_step, min_arrival_step, max_arrival_step, resolved_mode)
    end_hour_raw = _map_steps_to_hours(end_step, min_arrival_step, max_arrival_step, resolved_mode)
    deadline_hour = end_hour_raw.clip(0, 23)
    deadline_hour = np.maximum(deadline_hour, arrival_hour)
    waiting_time_steps = scheduled_start_step - arrival_step
    task_type = np.where(waiting_time_steps <= 1, "delay_sensitive", "delay_tolerant")
    priority = np.where(task_type == "delay_sensitive", 3, 2)

    task_id = _text_series(df, job_col, "", fill_counts, "task_id")
    empty_task_id = task_id.str.strip() == ""
    if empty_task_id.any():
        generated = pd.Series([f"real_task_{i}" for i in range(len(df))], index=df.index)
        task_id = task_id.where(~empty_task_id, generated)

    memory_demand = np.maximum(4.0, cpu_demand * 0.5)
    bandwidth_demand = pd.Series(0.5, index=df.index, dtype="float64")
    deadline_violation = end_hour_raw > deadline_hour
    random_marker = rng.random(len(df))

    mapped = pd.DataFrame(
        {
            "task_id": task_id,
            "arrival_step": arrival_step.round(4),
            "scheduled_start_step": scheduled_start_step.round(4),
            "execution_time_steps": execution_time_steps.round(4),
            "end_step": end_step.round(4),
            "arrival_time": arrival_hour.astype(int),
            "deadline": deadline_hour.astype(int),
            "task_type": task_type,
            "cpu_demand": cpu_demand.round(4),
            "memory_demand": pd.Series(memory_demand, index=df.index).round(4),
            "bandwidth_demand": bandwidth_demand,
            "priority": priority.astype(int),
            "datacenter_id": _text_series(df, dc_col, "unknown_dc", fill_counts, "datacenter_id"),
            "server_room_id": _text_series(df, room_col, "unknown_room", fill_counts, "server_room_id"),
            "rack_id": _text_series(df, rack_col, "unknown_rack", fill_counts, "rack_id"),
            "server_id": _text_series(df, server_col, "unknown_server", fill_counts, "server_id"),
            "global_server_id": _text_series(df, global_server_col, "unknown_global_server", fill_counts, "global_server_id"),
            "waiting_time_steps": waiting_time_steps.round(4),
            "deadline_violation": deadline_violation.astype(bool),
            "sample_marker": random_marker.round(8),
        }
    )

    task_24h = mapped[
        [
            "task_id",
            "arrival_time",
            "task_type",
            "cpu_demand",
            "memory_demand",
            "bandwidth_demand",
            "deadline",
            "priority",
            "arrival_step",
            "scheduled_start_step",
            "execution_time_steps",
            "end_step",
            "datacenter_id",
            "server_room_id",
            "rack_id",
            "server_id",
            "global_server_id",
            "deadline_violation",
        ]
    ].copy()

    save_csv(mapped, output_dir / "server_tasks_5min_raw_mapped.csv")
    save_csv(task_24h, output_dir / "server_tasks_24h.csv")

    required = ["task_id", "arrival_time", "task_type", "cpu_demand", "deadline", "server_room_id", "rack_id", "server_id"]
    missing = [col for col in required if col not in task_24h.columns]
    if missing:
        raise ValueError(f"server_tasks_24h.csv validation failed, missing columns: {missing}")
    if (task_24h["deadline"] < task_24h["arrival_time"]).any():
        raise ValueError("server_tasks_24h.csv validation failed: deadline is earlier than arrival_time.")

    meta = {
        "sheet_names": sheet_names,
        "sheet_name": sheet_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "warnings": warnings,
        "fill_counts": fill_counts,
        "time_mapping": time_mapping_meta,
    }
    return mapped, task_24h, meta


def clean_cluster_data(cluster_xlsx: Path, output_dir: Path, server_tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    warnings: list[str] = []
    fill_counts: dict[str, int] = {}
    workbook = _read_workbook(cluster_xlsx)
    try:
        sheet_name, df = _select_sheet(workbook, None, warnings)
        sheet_names = workbook.sheet_names
    finally:
        workbook.close()

    time_col = _find_column(df, ["时间步(5分钟)", "时间步（5分钟）", "time_step", "time step"], warnings)
    cap_col = _find_column(df, ["功率上限（标幺值）", "功率上限(标幺值)", "power_cap_pu", "power cap"], warnings)
    price_col = _find_column(df, ["电价（美元/兆瓦时）", "电价(美元/兆瓦时)", "price", "electricity price"], warnings)
    reward_col = _find_column(df, ["奖励值", "reward"], warnings)

    time_step = _numeric_series(df, time_col, 0.0, fill_counts, "time_step")
    if time_col is None:
        time_step = pd.Series(np.arange(len(df)), index=df.index, dtype="float64")
    power_cap_pu = _numeric_series(df, cap_col, 1.0, fill_counts, "power_cap_pu")
    price = _numeric_series(df, price_col, 0.0, fill_counts, "price")
    reward = _numeric_series(df, reward_col, 0.0, fill_counts, "reward")

    cluster = pd.DataFrame(
        {
            "time_step": time_step.astype(int),
            "power_cap_pu": power_cap_pu.round(6),
            "price": price.round(6),
            "reward": reward.round(6),
        }
    )
    cluster["hour"] = np.floor(cluster["time_step"] / 12).astype(int).clip(0, 23)
    save_csv(cluster[["time_step", "power_cap_pu", "price", "reward"]], output_dir / "cluster_power_price_5min.csv")

    hourly_cluster = (
        cluster.groupby("hour", as_index=False)
        .agg(price=("price", "mean"), power_limit=("power_cap_pu", "mean"))
        .set_index("hour")
        .reindex(range(24))
    )
    hourly_cluster["price"] = hourly_cluster["price"].interpolate(limit_direction="both").fillna(0.0)
    hourly_cluster["power_limit"] = hourly_cluster["power_limit"].interpolate(limit_direction="both").fillna(1.0)

    arrivals = server_tasks.groupby("arrival_time").size().reindex(range(24), fill_value=0)
    hourly_input = pd.DataFrame(
        {
            "hour": range(24),
            "arrival_rate": arrivals.values.astype(int),
            "price": hourly_cluster["price"].values.round(6),
            "carbon_factor": DEFAULT_CARBON,
            "power_limit": hourly_cluster["power_limit"].values.round(6),
        }
    )
    save_csv(hourly_input, output_dir / "hourly_input_24h.csv")

    meta = {
        "sheet_names": sheet_names,
        "sheet_name": sheet_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "warnings": warnings,
        "fill_counts": fill_counts,
    }
    return cluster[["time_step", "power_cap_pu", "price", "reward"]], hourly_input, meta


def clean_chip_data(chip_xlsx: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    warnings: list[str] = []
    fill_counts: dict[str, int] = {}
    workbook = _read_workbook(chip_xlsx)
    try:
        sheet_name, df = _select_sheet(workbook, CHIP_SHEET_NAME, warnings)
        sheet_names = workbook.sheet_names
    finally:
        workbook.close()

    actual_col = _find_column(df, ["实际功率(瓦)", "实际功率（瓦）", "actual_power_w"], warnings)
    reported_col = _find_column(df, ["报量功率（瓦）", "报量功率(瓦)", "reported_power_w"], warnings)
    cap_col = _find_column(df, ["功率上限（瓦）", "功率上限(瓦)", "power_cap_w"], warnings)
    freq_col = _find_column(df, ["频率(10^9Hz)", "频率（10^9Hz）", "frequency_ghz"], warnings)

    chip = pd.DataFrame(
        {
            "actual_power_w": _numeric_series(df, actual_col, 0.0, fill_counts, "actual_power_w").round(6),
            "reported_power_w": _numeric_series(df, reported_col, 0.0, fill_counts, "reported_power_w").round(6),
            "power_cap_w": _numeric_series(df, cap_col, 0.0, fill_counts, "power_cap_w").round(6),
            "frequency_ghz": _numeric_series(df, freq_col, 0.0, fill_counts, "frequency_ghz").round(6),
        }
    )
    save_csv(chip, output_dir / "chip_dvfs.csv")

    meta = {
        "sheet_names": sheet_names,
        "sheet_name": sheet_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "warnings": warnings,
        "fill_counts": fill_counts,
    }
    return chip, meta


def build_report(
    server_mapped: pd.DataFrame,
    server_tasks: pd.DataFrame,
    cluster: pd.DataFrame,
    hourly_input: pd.DataFrame,
    chip: pd.DataFrame,
    metas: dict[str, dict],
) -> list[str]:
    lines: list[str] = []
    lines.append("NBSDC uploaded case cleaning report")
    lines.append("=" * 44)
    lines.append("")
    for name, meta in metas.items():
        lines.append(f"[{name}]")
        lines.append(f"Sheets: {', '.join(meta['sheet_names'])}")
        lines.append(f"Selected sheet: {meta['sheet_name']}")
        lines.append(f"Raw shape: {meta['shape'][0]} rows x {meta['shape'][1]} columns")
        lines.append(f"Raw columns: {', '.join(map(str, meta['columns']))}")
        if meta["warnings"]:
            lines.append("Warnings:")
            for warning in meta["warnings"]:
                lines.append(f"- {warning}")
        lines.append("Missing/fill counts:")
        for key, value in sorted(meta["fill_counts"].items()):
            lines.append(f"- {key}: {value}")
        if name == "server" and "time_mapping" in meta:
            mapping = meta["time_mapping"]
            lines.append("Arrival time mapping:")
            lines.append(f"- arrival_step min: {mapping['arrival_step_min']:.4f}")
            lines.append(f"- arrival_step max: {mapping['arrival_step_max']:.4f}")
            lines.append(f"- requested mode: {mapping['arrival_time_mode_requested']}")
            lines.append(f"- resolved mode: {mapping['arrival_time_mode_resolved']}")
            lines.append(f"- rescale_24h enabled: {mapping['arrival_time_mode_resolved'] == 'rescale_24h'}")
            lines.append(f"- raw nonzero hours: {mapping['raw_nonzero_hours']}")
            lines.append(f"- raw hour>=8 share: {mapping['raw_late_hour_share']:.6f}")
        lines.append("")

    lines.append("Output files")
    lines.append("- data/real_case/cluster_power_price_5min.csv: time_step, power_cap_pu, price, reward")
    lines.append("- data/real_case/server_tasks_5min_raw_mapped.csv: mapped 5-minute server scheduling records")
    lines.append("- data/real_case/server_tasks_24h.csv: 24-hour task records with hierarchy fields")
    lines.append("- data/real_case/hourly_input_24h.csv: hour, arrival_rate, price, carbon_factor, power_limit")
    lines.append("- data/real_case/chip_dvfs.csv: actual_power_w, reported_power_w, power_cap_w, frequency_ghz")
    lines.append("")

    lines.append("Server task profile")
    lines.append(f"Server task total: {len(server_tasks)}")
    lines.append("arrival_time distribution:")
    for hour, count in server_tasks["arrival_time"].value_counts().sort_index().items():
        lines.append(f"- hour {int(hour):02d}: {int(count)}")
    lines.append("mapped hourly arrival distribution:")
    mapped_counts = server_tasks["arrival_time"].value_counts().reindex(range(24), fill_value=0).sort_index()
    for hour, count in mapped_counts.items():
        lines.append(f"- hour {int(hour):02d}: {int(count)}")
    lines.append("task_type distribution:")
    for task_type, count in server_tasks["task_type"].value_counts().items():
        lines.append(f"- {task_type}: {int(count)}")
    lines.append(
        "cpu_demand summary: "
        f"mean={server_tasks['cpu_demand'].mean():.4f}, "
        f"min={server_tasks['cpu_demand'].min():.4f}, "
        f"max={server_tasks['cpu_demand'].max():.4f}"
    )
    deadline_anomalies = int((server_tasks["deadline"] < server_tasks["arrival_time"]).sum())
    lines.append(f"deadline < arrival_time anomalies: {deadline_anomalies}")
    lines.append("")

    lines.append("Hourly input overview")
    lines.append(hourly_input[["hour", "arrival_rate", "price", "power_limit"]].to_string(index=False))
    lines.append("")

    lines.append("Chip DVFS profile")
    lines.append(f"chip_dvfs rows: {len(chip)}")
    lines.append(f"actual_power_w range: {chip['actual_power_w'].min():.4f} - {chip['actual_power_w'].max():.4f}")
    lines.append(f"frequency_ghz range: {chip['frequency_ghz'].min():.4f} - {chip['frequency_ghz'].max():.4f}")
    lines.append("")

    lines.append("Cluster 5-minute profile")
    lines.append(f"cluster_power_price rows: {len(cluster)}")
    lines.append(f"price range: {cluster['price'].min():.4f} - {cluster['price'].max():.4f}")
    lines.append(f"power_cap_pu range: {cluster['power_cap_pu'].min():.4f} - {cluster['power_cap_pu'].max():.4f}")
    lines.append("")
    lines.append("Validation: server_tasks_24h.csv passed required-column and deadline checks.")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean uploaded NBSDC XLSX files into project CSV inputs.")
    parser.add_argument("--cluster-xlsx", default=str(DEFAULT_CLUSTER_XLSX), help="Cluster-level XLSX path.")
    parser.add_argument("--server-xlsx", default=str(DEFAULT_SERVER_XLSX), help="Server-level XLSX path.")
    parser.add_argument("--chip-xlsx", default=str(DEFAULT_CHIP_XLSX), help="Chip-level XLSX path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for cleaned CSV files.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Cleaning report path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic helper fields.")
    parser.add_argument(
        "--arrival-time-mode",
        choices=["auto", "raw_step", "rescale_24h"],
        default="auto",
        help="Map server arrival steps to 24 hours using raw 5-minute steps, min-max rescaling, or automatic selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_xlsx = _project_path(args.cluster_xlsx)
    server_xlsx = _project_path(args.server_xlsx)
    chip_xlsx = _project_path(args.chip_xlsx)
    for path in [cluster_xlsx, server_xlsx, chip_xlsx]:
        if not path.exists():
            raise FileNotFoundError(f"Input XLSX not found: {path}")

    server_mapped, server_tasks, server_meta = clean_server_data(server_xlsx, output_dir, int(args.seed), args.arrival_time_mode)
    cluster, hourly_input, cluster_meta = clean_cluster_data(cluster_xlsx, output_dir, server_tasks)
    chip, chip_meta = clean_chip_data(chip_xlsx, output_dir)

    report = build_report(
        server_mapped=server_mapped,
        server_tasks=server_tasks,
        cluster=cluster,
        hourly_input=hourly_input,
        chip=chip,
        metas={"cluster": cluster_meta, "server": server_meta, "chip": chip_meta},
    )
    report_path = write_text(args.report_path, report)
    print(f"Cleaned uploaded case data into: {output_dir}")
    print(f"Cleaning report: {report_path}")


if __name__ == "__main__":
    main()
