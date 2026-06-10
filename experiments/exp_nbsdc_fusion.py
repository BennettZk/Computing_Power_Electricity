from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.io_utils import copy_file, copy_matching_files, ensure_dir, read_csv_required, save_csv, write_text
from utils.metrics import RESULT_CSV_COLUMN_MAPPING, export_csv_chinese, normalize_by_max, normalize_minmax
from utils.plotting import (
    PAPER_COLORS,
    apply_paper_axes,
    format_hour_axis,
    save_bar_plot,
    save_figure,
    save_line_plot,
    save_multi_line_plot,
    setup_chinese_matplotlib,
)


def _natural_sort_key(value):
    """生成自然排序键，使包含数字的名称按数值顺序排列。"""
    import re

    text = str(value)
    match = re.search(r"\d+", text)
    if match:
        return (0, int(match.group()), text)
    return (1, text)


def _hours() -> pd.DataFrame:
    """返回固定的 24 小时时间索引，统一小时级实验口径。"""
    return pd.DataFrame({"hour": range(24)})


def _derive_chip_hour(chip_df: pd.DataFrame) -> pd.DataFrame:
    """根据芯片级记录推导所属小时，用于对齐多层数据。"""
    chip = chip_df.copy()
    if "time_step" not in chip.columns:
        chip["time_step"] = np.arange(len(chip)) % 288
    chip["hour"] = np.floor(pd.to_numeric(chip["time_step"], errors="coerce").fillna(0) / 12).astype(int).clip(0, 23)
    return chip


def _cluster_hourly(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """聚合集群级原始数据，得到小时级电价和功率上限指标。"""
    cluster = cluster_df.copy()
    cluster["time_step"] = pd.to_numeric(cluster["time_step"], errors="coerce").fillna(0).astype(int)
    cluster["hour"] = np.floor(cluster["time_step"] / 12).astype(int).clip(0, 23)
    hourly = (
        cluster.groupby("hour", as_index=False)
        .agg(price=("price", "mean"), power_cap_pu=("power_cap_pu", "mean"))
        .rename(columns={"power_cap_pu": "hourly_power_cap"})
    )
    hourly = _hours().merge(hourly, on="hour", how="left")
    hourly["price"] = hourly["price"].interpolate(limit_direction="both").fillna(0.0)
    hourly["hourly_power_cap"] = hourly["hourly_power_cap"].interpolate(limit_direction="both").fillna(1.0)
    high_threshold = hourly["price"].quantile(0.75)
    low_threshold = hourly["price"].quantile(0.25)
    hourly["high_price_flag"] = (hourly["price"] >= high_threshold).astype(int)
    hourly["low_price_flag"] = (hourly["price"] <= low_threshold).astype(int)
    return hourly


def _server_hourly(server_df: pd.DataFrame) -> pd.DataFrame:
    """聚合服务器级调度数据，得到小时级任务和负载指标。"""
    server = server_df.copy()
    server["arrival_time"] = pd.to_numeric(server["arrival_time"], errors="coerce").fillna(0).astype(int).clip(0, 23)
    server["cpu_demand"] = pd.to_numeric(server["cpu_demand"], errors="coerce").fillna(0.0)
    if "deadline_violation" not in server.columns:
        server["deadline_violation"] = False
    server["deadline_violation"] = server["deadline_violation"].astype(str).str.lower().isin(["true", "1", "yes"])
    server["completed"] = (~server["deadline_violation"]).astype(int)
    hourly = (
        server.groupby("arrival_time", as_index=False)
        .agg(
            hourly_task_arrivals=("task_id", "count"),
            hourly_cpu_demand=("cpu_demand", "sum"),
            hourly_completed_tasks=("completed", "sum"),
            hourly_deadline_violation_rate=("deadline_violation", "mean"),
        )
        .rename(columns={"arrival_time": "hour"})
    )
    hourly = _hours().merge(hourly, on="hour", how="left").fillna(
        {
            "hourly_task_arrivals": 0,
            "hourly_cpu_demand": 0,
            "hourly_completed_tasks": 0,
            "hourly_deadline_violation_rate": 0,
        }
    )
    return hourly


def _chip_hourly(chip_df: pd.DataFrame) -> pd.DataFrame:
    """聚合芯片级数据，得到小时级实际功率、报量功率和频率指标。"""
    chip = _derive_chip_hour(chip_df)
    for col in ["actual_power_w", "reported_power_w", "power_cap_w", "frequency_ghz"]:
        chip[col] = pd.to_numeric(chip[col], errors="coerce").fillna(0.0)
    hourly = (
        chip.groupby("hour", as_index=False)
        .agg(
            hourly_actual_power_w=("actual_power_w", "mean"),
            hourly_reported_power_w=("reported_power_w", "mean"),
            hourly_power_cap_w=("power_cap_w", "mean"),
            hourly_frequency_ghz=("frequency_ghz", "mean"),
        )
    )
    hourly = _hours().merge(hourly, on="hour", how="left")
    for col in ["hourly_actual_power_w", "hourly_reported_power_w", "hourly_power_cap_w", "hourly_frequency_ghz"]:
        hourly[col] = hourly[col].interpolate(limit_direction="both").fillna(0.0)
    denominator = hourly["hourly_power_cap_w"].replace(0, np.nan)
    hourly["dvfs_tracking_error"] = ((hourly["hourly_actual_power_w"] - hourly["hourly_reported_power_w"]).abs() / denominator).fillna(0.0)
    return hourly


def _normalize_cluster_cap(power_cap: pd.Series) -> pd.Series:
    """将集群功率上限转换为归一化功率容量指标。"""
    values = pd.Series(power_cap, dtype="float64").fillna(0.0)
    if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        return values.clip(lower=0.0, upper=1.0)
    return normalize_minmax(values)


def _save_distribution_tables(server_df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """导出机房、机架和服务器层级的任务分布统计表。"""
    server = server_df.copy()
    server["cpu_demand"] = pd.to_numeric(server["cpu_demand"], errors="coerce").fillna(0.0)

    def distribution(column: str, label: str) -> pd.DataFrame:
        """按指定层级统计任务分布并补充层级名称。"""
        if column not in server.columns:
            return pd.DataFrame({label: ["unknown"], "task_count": [len(server)], "cpu_usage": [server["cpu_demand"].sum()]})
        dist = (
            server.groupby(column, as_index=False)
            .agg(task_count=("task_id", "count"), cpu_usage=("cpu_demand", "sum"))
            .rename(columns={column: label})
        )
        dist["_sort_key"] = dist[label].map(_natural_sort_key)
        return dist.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    room = distribution("server_room_id", "server_room_id")
    rack = distribution("rack_id", "rack_id")
    server_dist = distribution("server_id", "server_id")
    save_csv(room, output_dir / "room_task_distribution.csv")
    save_csv(rack, output_dir / "rack_task_distribution.csv")
    save_csv(server_dist, output_dir / "server_task_distribution.csv")
    export_csv_chinese(room, output_dir / "room_task_distribution_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    export_csv_chinese(rack, output_dir / "rack_task_distribution_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    export_csv_chinese(server_dist, output_dir / "server_task_distribution_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    return {"room": room, "rack": rack, "server": server_dist}


def _plot_price_powercap(aligned: pd.DataFrame, output_dir: Path) -> None:
    """绘制电价与功率上限的小时级对比图。"""
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    price_color = PAPER_COLORS["blue"]
    cap_color = PAPER_COLORS["green"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(aligned["hour"], aligned["price"], marker="o", linewidth=2.6, color=price_color, label="电价")
    ax1.set_xlabel("小时/h")
    ax1.set_ylabel("电价/(USD/MWh)", color=price_color)
    ax1.tick_params(axis="y", colors=price_color)
    apply_paper_axes(ax1)
    format_hour_axis(ax1)
    ax2 = ax1.twinx()
    ax2.plot(aligned["hour"], aligned["hourly_power_cap"], marker="s", linewidth=2.3, color=cap_color, label="功率上限")
    ax2.set_ylabel("功率上限/(p.u.)", color=cap_color)
    ax2.tick_params(axis="y", colors=cap_color)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax1.set_title("电价与功率上限关系")
    save_figure(fig, output_dir / "price_powercap.png")
    plt.close(fig)


def _plot_chip_actual_vs_cap(chip_hourly: pd.DataFrame, output_dir: Path) -> None:
    """绘制芯片实际功率与功率上限的对比图。"""
    save_multi_line_plot(
        chip_hourly,
        x="hour",
        y_columns=["hourly_actual_power_w", "hourly_power_cap_w"],
        path=output_dir / "chip_actual_vs_cap.png",
        title="芯片实际功率与功率上限对比",
        xlabel="小时/h",
        ylabel="功率/W",
        labels={"hourly_actual_power_w": "实际功率", "hourly_power_cap_w": "功率上限"},
    )


def _plot_dvfs_frequency_power(chip_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制 DVFS 频率与芯片功率响应关系图。"""
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    plot_df = chip_df[["frequency_ghz", "actual_power_w"]].copy()
    plot_df["frequency_ghz"] = pd.to_numeric(plot_df["frequency_ghz"], errors="coerce")
    plot_df["actual_power_w"] = pd.to_numeric(plot_df["actual_power_w"], errors="coerce")
    plot_df = plot_df.dropna().sort_values("frequency_ghz")
    output_path = output_dir / "dvfs_frequency_power.png"

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.set_title("DVFS频率与实际功率散点关系")
        ax.set_xlabel("频率/GHz")
        ax.set_ylabel("实际功率/W")
        apply_paper_axes(ax)
        save_figure(fig, output_path)
        plt.close(fig)
        return

    grouped = [(freq, group["actual_power_w"].to_numpy()) for freq, group in plot_df.groupby("frequency_ghz", sort=True)]
    has_repeated_frequency = len(grouped) > 1 and all(len(values) > 1 for _, values in grouped)
    caption_text = "图中展示 DVFS 频率档位与实际功率响应的相关关系。"

    if has_repeated_frequency:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        positions = [float(freq) for freq, _ in grouped]
        values = [power for _, power in grouped]
        min_gap = np.diff(sorted(positions)).min() if len(positions) > 1 else 0.1
        box_width = min(0.08, max(0.035, float(min_gap) * 0.42))
        box = ax.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PAPER_COLORS["red"], "linewidth": 2.3},
            boxprops={"edgecolor": PAPER_COLORS["blue"], "linewidth": 1.4},
            whiskerprops={"color": PAPER_COLORS["blue"], "linewidth": 1.2},
            capprops={"color": PAPER_COLORS["blue"], "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#bfdbfe")
            patch.set_alpha(0.55)

        jitter_rng = np.random.default_rng(42)
        max_points = 2200
        sample_df = plot_df.sample(n=min(len(plot_df), max_points), random_state=42) if len(plot_df) > max_points else plot_df
        jitter_width = box_width * 0.32
        jittered_x = sample_df["frequency_ghz"].to_numpy() + jitter_rng.uniform(-jitter_width, jitter_width, size=len(sample_df))
        ax.scatter(
            jittered_x,
            sample_df["actual_power_w"],
            s=12,
            color=PAPER_COLORS["gray"],
            alpha=0.32,
            edgecolors="none",
            zorder=3,
        )

        medians = [float(np.median(power)) for power in values]
        ax.scatter(positions, medians, marker="D", s=58, color=PAPER_COLORS["green"], edgecolors="white", linewidths=0.8, zorder=4)
        sample_label = f"抖动样本点（抽样 n={len(sample_df)}）" if len(sample_df) < len(plot_df) else f"抖动样本点（n={len(sample_df)}）"
        handles = [
            Patch(facecolor="#bfdbfe", edgecolor=PAPER_COLORS["blue"], alpha=0.55, label="功率分布箱线"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_COLORS["gray"], markeredgecolor="none", alpha=0.45, label=sample_label),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=PAPER_COLORS["green"], markeredgecolor="white", label="档位中位数"),
        ]
        ax.set_title("DVFS频率下实际功率分布")
        ax.legend(handles=handles, loc="upper left")
        caption_text = "箱线与抖动点展示同一频率档位下实际功率分布，用于说明 DVFS 档位与芯片功率响应的相关关系。"
    else:
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        ax.scatter(plot_df["frequency_ghz"], plot_df["actual_power_w"], s=22, color=PAPER_COLORS["blue"], alpha=0.55, edgecolors="none", label="样本点")
        r = float(np.corrcoef(plot_df["frequency_ghz"], plot_df["actual_power_w"])[0, 1]) if len(plot_df) >= 2 else float("nan")
        if len(plot_df) >= 2 and plot_df["frequency_ghz"].nunique() >= 2:
            slope, intercept = np.polyfit(plot_df["frequency_ghz"], plot_df["actual_power_w"], 1)
            x_line = np.linspace(float(plot_df["frequency_ghz"].min()), float(plot_df["frequency_ghz"].max()), 100)
            ax.plot(x_line, slope * x_line + intercept, linewidth=2.2, color=PAPER_COLORS["orange"], label="一阶趋势线")
        r_label = f"Pearson r = {r:.3f}" if np.isfinite(r) else "Pearson r = N/A"
        ax.text(
            0.03,
            0.95,
            r_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            color="#374151",
        )
        ax.set_title("DVFS频率与实际功率散点关系")
        ax.legend(loc="upper left")
        caption_text = "散点、一阶趋势线与 Pearson r 用于辅助观察 DVFS 频率与实际功率的相关关系。"

    ax.set_xlabel("频率/GHz")
    ax.set_ylabel("实际功率/W")
    ax.set_xticks([float(freq) for freq, _ in grouped])
    ax.set_xticklabels([f"{float(freq):.1f}" for freq, _ in grouped])
    if grouped:
        positions = [float(freq) for freq, _ in grouped]
        ax.set_xlim(min(positions) - 0.12, max(positions) + 0.12)
    apply_paper_axes(ax)
    fig.text(
        0.5,
        0.01,
        caption_text,
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(bottom=0.16)
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_room_hourly_task_heatmap(server_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制机房维度的小时任务量热力图。"""
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    server = server_df.copy()
    server["arrival_time"] = pd.to_numeric(server["arrival_time"], errors="coerce").fillna(0).astype(int).clip(0, 23)
    if "server_room_id" not in server.columns:
        server["server_room_id"] = "unknown_room"
    server["server_room_id"] = server["server_room_id"].astype(str)

    heatmap_df = (
        server.groupby(["server_room_id", "arrival_time"], as_index=False)
        .agg(task_count=("task_id", "count"))
        .pivot(index="server_room_id", columns="arrival_time", values="task_count")
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )
    ordered_index = sorted(heatmap_df.index.tolist(), key=_natural_sort_key)
    heatmap_df = heatmap_df.reindex(ordered_index)

    fig_height = max(3.8, min(8.0, 0.45 * max(len(heatmap_df), 1) + 2.2))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="Blues")
    ax.set_title("不同机房小时级任务分布")
    ax.set_xlabel("小时/h")
    ax.set_ylabel("机房")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(hour) for hour in range(24)])
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index.tolist())
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("任务数量/个")
    save_figure(fig, output_dir / "room_hourly_task_heatmap.png")
    plt.close(fig)


def run_nbsdc_fusion(
    data_dir: str | Path = "data/real_case",
    output_dir: str | Path | None = None,
    data_output_dir: str | Path = "outputs/data/nbsdc_fusion",
    figure_output_dir: str | Path = "outputs/figures/nbsdc_fusion",
    report_output_dir: str | Path = "outputs/reports",
    compat_output_dir: str | Path | None = "outputs/nbsdc_fusion",
    alpha: float = 0.55,
    beta: float = 0.25,
    base_reserve: float = 0.10,
) -> dict[str, Path]:
    """运行 NBSDC 三层数据融合实验并导出指标、图表和报告。"""
    data_dir = Path(data_dir)
    data_output_dir = ensure_dir(data_output_dir)
    figure_output_dir = ensure_dir(figure_output_dir)
    report_output_dir = ensure_dir(report_output_dir)
    compat_dir = ensure_dir(output_dir or compat_output_dir) if (output_dir or compat_output_dir) else None

    cluster = read_csv_required(data_dir / "cluster_power_price_5min.csv", ["time_step", "power_cap_pu", "price"])
    server_raw = read_csv_required(data_dir / "server_tasks_5min_raw_mapped.csv", ["task_id", "arrival_time", "cpu_demand"])
    server_24h = read_csv_required(data_dir / "server_tasks_24h.csv", ["task_id", "arrival_time", "cpu_demand", "deadline"])
    hourly_input = read_csv_required(data_dir / "hourly_input_24h.csv", ["hour", "arrival_rate", "price"])
    chip = read_csv_required(data_dir / "chip_dvfs.csv", ["actual_power_w", "reported_power_w", "power_cap_w", "frequency_ghz"])

    cluster_hourly = _cluster_hourly(cluster)
    server_hourly = _server_hourly(server_24h)
    chip_hourly = _chip_hourly(chip)
    distributions = _save_distribution_tables(server_24h, data_output_dir)

    aligned = (
        _hours()
        .merge(cluster_hourly, on="hour", how="left")
        .merge(server_hourly, on="hour", how="left")
        .merge(chip_hourly, on="hour", how="left")
        .merge(hourly_input[["hour", "arrival_rate"]], on="hour", how="left")
    )
    old_cluster_cap_norm = normalize_by_max(aligned["hourly_power_cap"])
    old_server_load_norm = normalize_by_max(aligned["hourly_cpu_demand"])
    aligned["chip_power_norm"] = normalize_by_max(aligned["hourly_actual_power_w"])
    aligned["power_margin_norm_old"] = (
        old_cluster_cap_norm - 0.5 * old_server_load_norm - 0.5 * aligned["chip_power_norm"]
    ).clip(lower=0.0, upper=1.0)

    aligned["cluster_cap_norm"] = _normalize_cluster_cap(aligned["hourly_power_cap"])
    aligned["server_load_norm"] = normalize_minmax(aligned["hourly_cpu_demand"])
    aligned["chip_power_variation_norm"] = normalize_minmax(aligned["hourly_actual_power_w"])
    aligned["chip_power_ratio"] = (
        aligned["hourly_actual_power_w"] / aligned["hourly_power_cap_w"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aligned["power_margin_norm_v2"] = (
        aligned["cluster_cap_norm"]
        - alpha * aligned["server_load_norm"]
        - beta * aligned["chip_power_variation_norm"]
        - float(base_reserve)
    ).clip(lower=0.0, upper=1.0)
    aligned["power_margin_norm"] = aligned["power_margin_norm_v2"]
    aligned["alpha"] = float(alpha)
    aligned["beta"] = float(beta)
    aligned["base_reserve"] = float(base_reserve)

    aligned_path = save_csv(aligned, data_output_dir / "aligned_hourly_fusion.csv")
    export_csv_chinese(aligned, data_output_dir / "aligned_hourly_fusion_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    metrics_cols = [
        "hour",
        "price",
        "hourly_power_cap",
        "high_price_flag",
        "low_price_flag",
        "hourly_task_arrivals",
        "hourly_cpu_demand",
        "hourly_completed_tasks",
        "hourly_deadline_violation_rate",
        "hourly_actual_power_w",
        "hourly_reported_power_w",
        "hourly_power_cap_w",
        "hourly_frequency_ghz",
        "dvfs_tracking_error",
        "cluster_cap_norm",
        "server_load_norm",
        "chip_power_norm",
        "chip_power_variation_norm",
        "chip_power_ratio",
        "power_margin_norm_old",
        "power_margin_norm_v2",
        "power_margin_norm",
    ]
    metrics_df = aligned[metrics_cols]
    metrics_path = save_csv(metrics_df, data_output_dir / "nbsdc_fusion_metrics.csv")
    export_csv_chinese(metrics_df, data_output_dir / "nbsdc_fusion_metrics_cn.csv", RESULT_CSV_COLUMN_MAPPING)

    _plot_price_powercap(aligned, figure_output_dir)
    save_line_plot(
        aligned,
        x="hour",
        y="hourly_task_arrivals",
        path=figure_output_dir / "hourly_task_arrivals.png",
        title="服务器级24小时任务到达量",
        xlabel="小时/h",
        ylabel="任务到达量/个",
    )
    _plot_dvfs_frequency_power(chip, figure_output_dir)
    save_multi_line_plot(
        aligned,
        x="hour",
        y_columns=["cluster_cap_norm", "server_load_norm", "chip_power_variation_norm", "power_margin_norm"],
        path=figure_output_dir / "three_layer_power_margin.png",
        title="三层数据融合下的等效功率裕度",
        xlabel="小时/h",
        ylabel="归一化值",
        labels={
            "cluster_cap_norm": "集群功率上限",
            "server_load_norm": "服务器负载",
            "chip_power_variation_norm": "芯片功率波动",
            "power_margin_norm": "等效功率裕度",
        },
    )
    save_multi_line_plot(
        aligned,
        x="hour",
        y_columns=["power_margin_norm_old", "power_margin_norm"],
        path=figure_output_dir / "power_margin_baseline_vs_fused.png",
        title="基准裕度与三层融合裕度对比",
        xlabel="小时/h",
        ylabel="等效功率裕度",
        labels={"power_margin_norm_old": "基准裕度", "power_margin_norm": "三层融合裕度"},
    )
    stale_old_plot = figure_output_dir / "power_margin_old_vs_new.png"
    if stale_old_plot.exists():
        stale_old_plot.unlink()
    if compat_dir:
        compat_stale_old_plot = compat_dir / "power_margin_old_vs_new.png"
        if compat_stale_old_plot.exists():
            compat_stale_old_plot.unlink()

    _plot_room_hourly_task_heatmap(server_24h, figure_output_dir)
    room_top = distributions["room"].head(12).copy()
    save_bar_plot(
        room_top,
        x="server_room_id",
        y="task_count",
        path=figure_output_dir / "room_task_distribution.png",
        title="不同机房任务总量分布（辅助）",
        xlabel="机房",
        ylabel="任务数量/个",
        rotation=30,
    )
    _plot_chip_actual_vs_cap(chip_hourly, figure_output_dir)

    summary_lines = [
        "NBSDC三层数据融合摘要",
        "=" * 28,
        f"集群级记录数: {len(cluster)}",
        f"服务器级原始映射记录数: {len(server_raw)}",
        f"服务器级24小时任务记录数: {len(server_24h)}",
        f"芯片级DVFS记录数: {len(chip)}",
        f"平均电价: {aligned['price'].mean():.4f}",
        f"平均小时任务到达量: {aligned['hourly_task_arrivals'].mean():.2f}",
        f"芯片平均实际功率: {aligned['hourly_actual_power_w'].mean():.4f} W",
        f"芯片功率波动归一化均值: {aligned['chip_power_variation_norm'].mean():.4f}",
        f"芯片实际功率/功率上限均值: {aligned['chip_power_ratio'].mean():.4f}",
        f"基准等效功率裕度均值: {aligned['power_margin_norm_old'].mean():.4f}",
        f"三层融合等效功率裕度均值: {aligned['power_margin_norm'].mean():.4f}",
        f"三层融合等效功率裕度最大值: {aligned['power_margin_norm'].max():.4f}",
        f"alpha: {alpha}",
        f"beta: {beta}",
        f"base_reserve: {base_reserve}",
        "",
        "等效功率裕度由集群功率上限、服务器负载和芯片功率响应三层指标融合得到。",
        "由于芯片实际功率存在较高基础功耗，本文采用 min-max 归一化刻画芯片功率的相对波动，避免最大值归一化导致曲线过平。",
    ]
    summary_path = write_text(report_output_dir / "nbsdc_fusion_summary.txt", summary_lines)

    if compat_dir:
        copy_matching_files(data_output_dir, compat_dir, [".csv"])
        copy_matching_files(figure_output_dir, compat_dir, [".png", ".pdf"])
        copy_file(summary_path, compat_dir / summary_path.name)

    return {
        "aligned_hourly_fusion": aligned_path,
        "nbsdc_fusion_metrics": metrics_path,
        "summary": summary_path,
    }
