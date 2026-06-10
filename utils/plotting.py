from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures_final"

PAPER_COLORS = {
    "blue": "#2563eb",
    "green": "#15803d",
    "orange": "#ea580c",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#64748b",
}
PAPER_PALETTE = [
    PAPER_COLORS["blue"],
    PAPER_COLORS["green"],
    PAPER_COLORS["orange"],
    PAPER_COLORS["red"],
    PAPER_COLORS["purple"],
    PAPER_COLORS["gray"],
]
HIGHLIGHT_COLUMNS = {
    "power_margin_norm",
    "net_token_profit",
    "RH-TEO",
    "RH-TEO策略",
    "Token净收益",
    "等效功率裕度",
}


def setup_chinese_matplotlib() -> None:
    """优先使用常见 Windows 中文字体，并以无界面方式保存图表。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#9ca3af"
    plt.rcParams["axes.labelcolor"] = "#111827"
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.color"] = "#374151"
    plt.rcParams["ytick.color"] = "#374151"
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.95
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "#d1d5db"


def _resolve_path(path: str | Path) -> Path:
    """把输出路径解析为项目内绝对路径。"""
    output_path = Path(path)
    return output_path if output_path.is_absolute() else PROJECT_ROOT / output_path


def apply_paper_axes(ax, grid_axis: str = "both") -> None:
    """统一设置论文图表坐标轴和网格样式。"""
    ax.grid(True, axis=grid_axis, color="#e5e7eb", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def save_figure(fig, path: str | Path) -> None:
    """同时保存 PNG 和 PDF 图表，并同步到最终图表目录。"""
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, FINAL_FIGURE_DIR / png_path.name)
    shutil.copy2(pdf_path, FINAL_FIGURE_DIR / pdf_path.name)


def format_hour_axis(ax) -> None:
    """设置小时轴范围和刻度。"""
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))


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
    """绘制并保存单条折线图。"""
    setup_chinese_matplotlib()
    output_path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(df[x], df[y], marker=marker, linewidth=2.8, color=PAPER_COLORS["blue"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_paper_axes(ax)
    if x == "hour":
        format_hour_axis(ax)
    save_figure(fig, output_path)
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
    """绘制并保存多条折线对比图。"""
    setup_chinese_matplotlib()
    output_path = Path(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, col in enumerate(y_columns):
        label = (labels or {}).get(col, col)
        is_highlight = col in HIGHLIGHT_COLUMNS or label in HIGHLIGHT_COLUMNS
        ax.plot(
            df[x],
            df[col],
            marker="o",
            linewidth=3.2 if is_highlight else 1.9,
            markersize=5.2 if is_highlight else 4.2,
            color=PAPER_PALETTE[idx % len(PAPER_PALETTE)],
            alpha=1.0 if is_highlight else 0.82,
            label=label,
            zorder=3 if is_highlight else 2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_paper_axes(ax)
    if x == "hour":
        format_hour_axis(ax)
    ax.legend(ncol=2 if len(y_columns) > 3 else 1)
    save_figure(fig, output_path)
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
    """绘制并保存柱状图。"""
    setup_chinese_matplotlib()
    output_path = Path(path)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(df[x].astype(str), df[y], color=PAPER_COLORS["blue"], alpha=0.88)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    apply_paper_axes(ax, grid_axis="y")
    save_figure(fig, output_path)
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
    """绘制并保存散点图。"""
    setup_chinese_matplotlib()
    output_path = Path(path)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df[x], df[y], s=14, alpha=alpha, color=PAPER_COLORS["blue"], edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_paper_axes(ax)
    save_figure(fig, output_path)
    plt.close(fig)
