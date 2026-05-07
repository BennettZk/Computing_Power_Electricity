from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.io_utils import ensure_dir, read_csv_required, save_csv, write_text
from utils.metrics import normalize_by_max
from utils.plotting import save_bar_plot, save_line_plot, save_multi_line_plot, save_scatter_plot, setup_chinese_matplotlib


def _hours() -> pd.DataFrame:
    return pd.DataFrame({"hour": range(24)})


def _derive_chip_hour(chip_df: pd.DataFrame) -> pd.DataFrame:
    chip = chip_df.copy()
    if "time_step" not in chip.columns:
        chip["time_step"] = np.arange(len(chip)) % 288
    chip["hour"] = np.floor(pd.to_numeric(chip["time_step"], errors="coerce").fillna(0) / 12).astype(int).clip(0, 23)
    return chip


def _cluster_hourly(cluster_df: pd.DataFrame) -> pd.DataFrame:
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


def _save_distribution_tables(server_df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    server = server_df.copy()
    server["cpu_demand"] = pd.to_numeric(server["cpu_demand"], errors="coerce").fillna(0.0)

    def distribution(column: str, label: str) -> pd.DataFrame:
        if column not in server.columns:
            return pd.DataFrame({label: ["unknown"], "task_count": [len(server)], "cpu_usage": [server["cpu_demand"].sum()]})
        return (
            server.groupby(column, as_index=False)
            .agg(task_count=("task_id", "count"), cpu_usage=("cpu_demand", "sum"))
            .rename(columns={column: label})
            .sort_values("task_count", ascending=False)
        )

    room = distribution("server_room_id", "server_room_id")
    rack = distribution("rack_id", "rack_id")
    server_dist = distribution("server_id", "server_id")
    save_csv(room, output_dir / "room_task_distribution.csv")
    save_csv(rack, output_dir / "rack_task_distribution.csv")
    save_csv(server_dist, output_dir / "server_task_distribution.csv")
    return {"room": room, "rack": rack, "server": server_dist}


def _plot_price_powercap(aligned: pd.DataFrame, output_dir: Path) -> None:
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(aligned["hour"], aligned["price"], marker="o", linewidth=2, color="#2563eb", label="电价")
    ax1.set_xlabel("小时")
    ax1.set_ylabel("电价")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(aligned["hour"], aligned["hourly_power_cap"], marker="s", linewidth=2, color="#dc2626", label="功率上限")
    ax2.set_ylabel("功率上限")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("电价与功率上限关系")
    fig.tight_layout()
    fig.savefig(output_dir / "price_powercap.png")
    plt.close(fig)


def _plot_chip_actual_vs_cap(chip_hourly: pd.DataFrame, output_dir: Path) -> None:
    save_multi_line_plot(
        chip_hourly,
        x="hour",
        y_columns=["hourly_actual_power_w", "hourly_power_cap_w"],
        path=output_dir / "chip_actual_vs_cap.png",
        title="芯片实际功率与功率上限对比",
        xlabel="小时",
        ylabel="功率/W",
        labels={"hourly_actual_power_w": "实际功率", "hourly_power_cap_w": "功率上限"},
    )


def run_nbsdc_fusion(
    data_dir: str | Path = "data/real_case",
    output_dir: str | Path = "outputs/nbsdc_fusion",
    alpha: float = 0.5,
    beta: float = 0.5,
) -> dict[str, Path]:
    data_dir = Path(data_dir)
    output_dir = ensure_dir(output_dir)

    cluster = read_csv_required(data_dir / "cluster_power_price_5min.csv", ["time_step", "power_cap_pu", "price"])
    server_raw = read_csv_required(data_dir / "server_tasks_5min_raw_mapped.csv", ["task_id", "arrival_time", "cpu_demand"])
    server_24h = read_csv_required(data_dir / "server_tasks_24h.csv", ["task_id", "arrival_time", "cpu_demand", "deadline"])
    hourly_input = read_csv_required(data_dir / "hourly_input_24h.csv", ["hour", "arrival_rate", "price"])
    chip = read_csv_required(data_dir / "chip_dvfs.csv", ["actual_power_w", "reported_power_w", "power_cap_w", "frequency_ghz"])

    cluster_hourly = _cluster_hourly(cluster)
    server_hourly = _server_hourly(server_24h)
    chip_hourly = _chip_hourly(chip)
    distributions = _save_distribution_tables(server_24h, output_dir)

    aligned = (
        _hours()
        .merge(cluster_hourly, on="hour", how="left")
        .merge(server_hourly, on="hour", how="left")
        .merge(chip_hourly, on="hour", how="left")
        .merge(hourly_input[["hour", "arrival_rate"]], on="hour", how="left")
    )
    aligned["cluster_cap_norm"] = normalize_by_max(aligned["hourly_power_cap"])
    aligned["server_load_norm"] = normalize_by_max(aligned["hourly_cpu_demand"])
    aligned["chip_power_norm"] = normalize_by_max(aligned["hourly_actual_power_w"])
    aligned["power_margin_norm"] = (
        aligned["cluster_cap_norm"] - alpha * aligned["server_load_norm"] - beta * aligned["chip_power_norm"]
    ).clip(lower=0.0, upper=1.0)
    aligned["alpha"] = float(alpha)
    aligned["beta"] = float(beta)

    aligned_path = save_csv(aligned, output_dir / "aligned_hourly_fusion.csv")
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
        "power_margin_norm",
    ]
    metrics_path = save_csv(aligned[metrics_cols], output_dir / "nbsdc_fusion_metrics.csv")

    _plot_price_powercap(aligned, output_dir)
    save_line_plot(
        aligned,
        x="hour",
        y="hourly_task_arrivals",
        path=output_dir / "hourly_task_arrivals.png",
        title="服务器级24小时任务到达量",
        xlabel="小时",
        ylabel="任务到达量",
    )
    chip_scatter = _derive_chip_hour(chip).sample(n=min(len(chip), 3000), random_state=42) if len(chip) else chip
    save_scatter_plot(
        chip_scatter,
        x="frequency_ghz",
        y="actual_power_w",
        path=output_dir / "dvfs_frequency_power.png",
        title="DVFS频率与实际功率关系",
        xlabel="频率/GHz",
        ylabel="实际功率/W",
    )
    save_multi_line_plot(
        aligned,
        x="hour",
        y_columns=["cluster_cap_norm", "server_load_norm", "chip_power_norm", "power_margin_norm"],
        path=output_dir / "three_layer_power_margin.png",
        title="三层数据融合下的等效功率裕度",
        xlabel="小时",
        ylabel="归一化值",
        labels={
            "cluster_cap_norm": "集群功率上限",
            "server_load_norm": "服务器负载",
            "chip_power_norm": "芯片功率",
            "power_margin_norm": "等效功率裕度",
        },
    )
    room_top = distributions["room"].head(12).copy()
    save_bar_plot(
        room_top,
        x="server_room_id",
        y="task_count",
        path=output_dir / "room_task_distribution.png",
        title="不同机房任务分布",
        xlabel="机房",
        ylabel="任务数量",
        rotation=30,
    )
    _plot_chip_actual_vs_cap(chip_hourly, output_dir)

    summary_lines = [
        "NBSDC three-layer fusion summary",
        "=" * 38,
        f"Cluster records: {len(cluster)}",
        f"Server raw records: {len(server_raw)}",
        f"Server 24h task records: {len(server_24h)}",
        f"Chip DVFS records: {len(chip)}",
        f"Average price: {aligned['price'].mean():.4f}",
        f"Average hourly task arrivals: {aligned['hourly_task_arrivals'].mean():.2f}",
        f"Average actual chip power: {aligned['hourly_actual_power_w'].mean():.4f} W",
        f"Average power margin norm: {aligned['power_margin_norm'].mean():.4f}",
        f"Max power margin norm: {aligned['power_margin_norm'].max():.4f}",
        f"alpha: {alpha}",
        f"beta: {beta}",
        "",
        "The fused power margin is a normalized scenario variable derived from cluster power cap, server load, and chip power response.",
    ]
    summary_path = write_text(output_dir / "nbsdc_fusion_summary.txt", summary_lines)

    return {
        "aligned_hourly_fusion": aligned_path,
        "nbsdc_fusion_metrics": metrics_path,
        "summary": summary_path,
    }
