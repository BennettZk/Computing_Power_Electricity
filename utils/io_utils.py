from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """解析相对于项目根目录的路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """创建目录并以 Path 对象返回。"""
    output_dir = project_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_yaml_like(path: str | Path) -> dict:
    """
    加载项目使用的轻量级、兼容 JSON 语法的 YAML 配置文件。

    项目在 README 中保留 .yaml 后缀以提升可读性，但配置语法有意保持
    JSON 兼容，从而避免引入 PyYAML 这类额外依赖。
    """
    config_path = project_path(path)
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON-compatible YAML: {config_path}") from exc


def write_text(path: str | Path, lines: str | Iterable[str]) -> Path:
    """写入 UTF-8 文本，支持传入字符串或多行列表。"""
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(str(line) for line in lines)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def read_csv_required(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """读取 CSV，并在缺少必需列时抛出清晰错误。"""
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
    """保存程序可读的 CSV，列名保持英文。"""
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """复制生成文件，并保留文件元数据。"""
    src_path = project_path(src)
    dst_path = project_path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def copy_matching_files(source_dir: str | Path, target_dir: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """把一个输出目录中指定后缀的文件复制到另一个目录。"""
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
