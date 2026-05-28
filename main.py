from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.exp_nbsdc_fusion import run_nbsdc_fusion
from experiments.exp_token_export import run_token_export
from scripts.prepare_uploaded_case_dataset import (
    CHIP_SHEET_NAME,
    DEFAULT_REPORT_PATH,
    SERVER_SHEET_NAME,
    SimpleXlsxWorkbook,
    build_report,
    clean_chip_data,
    clean_cluster_data,
    clean_server_data,
)
from utils.io_utils import ensure_dir, load_yaml_like, project_path, write_text


RAW_DIR = project_path("data/raw")
CLEANED_DATA_DIR = project_path("data/real_case")
TOKEN_CONFIG_PATH = project_path("config/token_export.yaml")


@dataclass(frozen=True)
class RawInputFiles:
    cluster: Path
    server: Path
    chip: Path


def _xlsx_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {raw_dir}")
    return sorted(files, key=lambda path: path.name.lower())


def _roles_from_filename(path: Path) -> set[str]:
    name = path.stem.casefold()
    roles: set[str] = set()
    if any(keyword in name for keyword in ["cluster", "datacenter", "data_center", "数据中心", "集群"]):
        roles.add("cluster")
    if any(keyword in name for keyword in ["server", "服务器", "服务端"]):
        roles.add("server")
    if any(keyword in name for keyword in ["chip", "dvfs", "芯片"]):
        roles.add("chip")
    return roles


def _role_from_workbook(path: Path) -> str | None:
    workbook = SimpleXlsxWorkbook(path)
    try:
        sheet_names = set(workbook.sheet_names)
    finally:
        workbook.close()
    if SERVER_SHEET_NAME in sheet_names:
        return "server"
    if CHIP_SHEET_NAME in sheet_names:
        return "chip"
    return None


def _single_candidate(role: str, candidates: list[Path], files: list[Path], raw_dir: Path) -> Path:
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        names = ", ".join(path.name for path in unique)
        raise ValueError(f"Multiple {role} raw files found: {names}")

    available = ", ".join(path.name for path in files)
    raise ValueError(
        f"Could not identify the {role} raw file in {raw_dir}. "
        f"Available files: {available}. "
        "Use filenames containing cluster/datacenter, server, and chip, "
        "or keep the original Chinese filenames."
    )


def discover_raw_inputs(raw_dir: str | Path = RAW_DIR) -> RawInputFiles:
    raw_dir = project_path(raw_dir)
    files = _xlsx_files(raw_dir)
    candidates: dict[str, list[Path]] = {"cluster": [], "server": [], "chip": []}
    unresolved: list[Path] = []

    for path in files:
        roles = _roles_from_filename(path)
        if len(roles) == 1:
            candidates[next(iter(roles))].append(path)
        else:
            unresolved.append(path)

    for path in unresolved:
        try:
            role = _role_from_workbook(path)
        except Exception:
            role = None
        if role:
            candidates[role].append(path)

    assigned = {path for paths in candidates.values() for path in paths}
    remaining = [path for path in files if path not in assigned]
    if not candidates["cluster"] and len(remaining) == 1:
        candidates["cluster"].append(remaining[0])

    return RawInputFiles(
        cluster=_single_candidate("cluster", candidates["cluster"], files, raw_dir),
        server=_single_candidate("server", candidates["server"], files, raw_dir),
        chip=_single_candidate("chip", candidates["chip"], files, raw_dir),
    )


def prepare_cleaned_data(raw_inputs: RawInputFiles, seed: int, arrival_time_mode: str) -> dict[str, Path]:
    output_dir = ensure_dir(CLEANED_DATA_DIR)
    server_mapped, server_tasks, server_meta = clean_server_data(
        raw_inputs.server,
        output_dir,
        seed=seed,
        arrival_time_mode=arrival_time_mode,
    )
    cluster, hourly_input, cluster_meta = clean_cluster_data(raw_inputs.cluster, output_dir, server_tasks)
    chip, chip_meta = clean_chip_data(raw_inputs.chip, output_dir)
    report = build_report(
        server_mapped=server_mapped,
        server_tasks=server_tasks,
        cluster=cluster,
        hourly_input=hourly_input,
        chip=chip,
        metas={"cluster": cluster_meta, "server": server_meta, "chip": chip_meta},
    )
    report_path = write_text(DEFAULT_REPORT_PATH, report)
    return {
        "cleaned_data_dir": output_dir,
        "cleaning_report": report_path,
    }


def run_pipeline(raw_dir: str | Path, seed: int, arrival_time_mode: str) -> dict[str, dict[str, Path]]:
    raw_inputs = discover_raw_inputs(raw_dir)
    print("Raw input files:")
    print(f"- cluster: {raw_inputs.cluster}")
    print(f"- server: {raw_inputs.server}")
    print(f"- chip: {raw_inputs.chip}")

    print("\n[1/3] Cleaning raw data...")
    cleaned_outputs = prepare_cleaned_data(raw_inputs, seed=seed, arrival_time_mode=arrival_time_mode)

    print("[2/3] Running NBSDC fusion...")
    fusion_outputs = run_nbsdc_fusion(data_dir=CLEANED_DATA_DIR)

    print("[3/3] Running token export optimization...")
    config = load_yaml_like(TOKEN_CONFIG_PATH)
    token_outputs = run_token_export(
        fusion_path=fusion_outputs["aligned_hourly_fusion"],
        config=config,
    )

    return {
        "cleaned": cleaned_outputs,
        "fusion": fusion_outputs,
        "token_export": token_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full NBSDC pipeline from data/raw XLSX files to final outputs."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw cluster/server/chip XLSX files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic helper fields.")
    parser.add_argument(
        "--arrival-time-mode",
        choices=["auto", "raw_step", "rescale_24h"],
        default="auto",
        help="Map server arrival steps to 24 hours.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        raw_dir=args.raw_dir,
        seed=int(args.seed),
        arrival_time_mode=args.arrival_time_mode,
    )

    print("\nPipeline finished.")
    print("Key outputs:")
    for group_name, group_outputs in outputs.items():
        print(f"[{group_name}]")
        for name, path in group_outputs.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
