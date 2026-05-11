from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    output_dir = project_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_yaml_like(path: str | Path) -> dict:
    """
    Load the lightweight JSON-compatible YAML files used by this project.

    The project keeps the .yaml suffix for readability in the README, but the
    configuration syntax is intentionally JSON-compatible to avoid an extra
    dependency such as PyYAML.
    """
    config_path = project_path(path)
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON-compatible YAML: {config_path}") from exc


def write_text(path: str | Path, lines: str | Iterable[str]) -> Path:
    """Write UTF-8 text, accepting either a string or a list of lines."""
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(str(line) for line in lines)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def read_csv_required(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV and raise a clear error if required columns are missing."""
    csv_path = project_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a program-readable CSV with English column names."""
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copy a generated file while preserving metadata."""
    src_path = project_path(src)
    dst_path = project_path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def copy_matching_files(source_dir: str | Path, target_dir: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """Copy files with selected suffixes from one output directory to another."""
    source_path = project_path(source_dir)
    target_path = ensure_dir(target_dir)
    allowed = {suffix.lower() for suffix in suffixes}
    copied: list[Path] = []
    if not source_path.exists():
        return copied
    for item in source_path.iterdir():
        if item.is_file() and item.suffix.lower() in allowed:
            copied.append(copy_file(item, target_path / item.name))
    return copied
