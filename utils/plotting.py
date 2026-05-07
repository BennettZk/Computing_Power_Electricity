from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def setup_chinese_matplotlib() -> None:
    """Use common Windows Chinese fonts when available and save figures headlessly."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    marker: str | None = "o",
) -> None:
    setup_chinese_matplotlib()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(df[x], df[y], marker=marker, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_multi_line_plot(
    df: pd.DataFrame,
    x: str,
    y_columns: list[str],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: dict[str, str] | None = None,
) -> None:
    setup_chinese_matplotlib()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in y_columns:
        ax.plot(df[x], df[col], marker="o", linewidth=2, label=(labels or {}).get(col, col))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 0,
) -> None:
    setup_chinese_matplotlib()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    alpha: float = 0.45,
) -> None:
    setup_chinese_matplotlib()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df[x], df[y], s=12, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
