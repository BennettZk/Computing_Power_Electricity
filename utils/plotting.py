from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def _configure_plot_fonts() -> None:
    """优先使用系统中文字体，避免中文场景名在图片中显示为方框。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_plot_fonts()


def plot_price_load_curve(hourly_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(hourly_df["hour"], hourly_df["price"], color="#b94700", marker="o", label="Electricity Price")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(hourly_df["hour"], hourly_df["arrival_rate"], alpha=0.25, color="#3a6ea5", label="Task Arrivals")
    ax2.set_ylabel("Hourly Arrivals")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pareto_front(pareto_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        pareto_df["total_cost"],
        pareto_df["avg_delay_hours"],
        c=pareto_df["load_imbalance"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("Total Cost")
    ax.set_ylabel("Average Delay (hours)")
    ax.set_title("Proposed Scheduler Pareto Front")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Load Imbalance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_total_energy_bar(results_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["algorithm"], results_df["total_energy_kwh"])
    ax.set_ylabel("Total Energy (kWh)")
    ax.set_title("Algorithm Energy Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cpu_gpu_utilization(results_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(results_df))
    width = 0.35
    ax.bar([value - width / 2 for value in x], results_df["avg_cpu_utilization"], width=width, label="CPU Utilization")
    ax.bar([value + width / 2 for value in x], results_df["avg_gpu_utilization"], width=width, label="GPU Utilization")
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["algorithm"])
    ax.set_ylabel("Average Utilization")
    ax.set_title("CPU / GPU Utilization Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_convergence_curve(convergence_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(convergence_df["generation"], convergence_df["best_total_cost"], label="Best Cost Objective")
    ax.plot(convergence_df["generation"], convergence_df["best_delay_objective"], label="Best Delay Objective")
    ax.plot(convergence_df["generation"], convergence_df["best_balance_objective"], label="Best Balance Objective")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective Value")
    ax.set_title("NSGA-II Convergence Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_objective_convergence(convergence_df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    if convergence_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(convergence_df["generation"], convergence_df["best_fitness"], label="Best Fitness")
    if "avg_fitness" in convergence_df.columns:
        ax.plot(convergence_df["generation"], convergence_df["avg_fitness"], label="Average Fitness", alpha=0.75)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ablation_results(ablation_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_labels = ["Full", "No Spatial", "No Time", "No Hetero", "No Priority"][: len(ablation_df)]
    ax1.bar(x_labels, ablation_df["total_cost"], color="#2f6f73", alpha=0.85, label="Total Cost")
    ax1.set_ylabel("Total Cost")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, ablation_df["avg_delay_hours"], color="#c2410c", marker="o", label="Average Delay")
    ax2.set_ylabel("Average Delay (hours)")

    fig.suptitle("Ablation Study: Cost and Delay")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sensitivity_results(sensitivity_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if "scenario" in sensitivity_df.columns:
        x_labels = sensitivity_df["scenario"].astype(str).tolist()
    else:
        x_labels = [f"Scenario {idx + 1}" for idx in range(len(sensitivity_df))]
    x = list(range(len(sensitivity_df)))

    ax1.plot(x, sensitivity_df["total_cost"], marker="o", color="#0b7285", label="Total Cost")
    ax1.set_ylabel("Total Cost")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis="x", rotation=18)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, sensitivity_df["sla_violation_rate"], marker="s", color="#b42318", label="SLA Violation")
    ax2.set_ylabel("SLA Violation Rate")

    fig.suptitle("Sensitivity Study: Cost and SLA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cross_region_delay_sensitivity(sensitivity_df: pd.DataFrame, output_path: str | Path) -> None:
    scenario_order = ["低跨区时延", "中跨区时延", "高跨区时延"]
    selected = sensitivity_df[sensitivity_df["scenario"].isin(scenario_order)].copy()
    if selected.empty:
        return

    selected["scenario"] = pd.Categorical(selected["scenario"], categories=scenario_order, ordered=True)
    selected = selected.sort_values("scenario")
    x_labels = selected["scenario"].astype(str).tolist()
    x = list(range(len(selected)))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, selected["remote_task_count"], color="#3b5bdb", alpha=0.85, label="Remote Task Count")
    ax1.set_xlabel("跨区时延场景")
    ax1.set_ylabel("Remote Task Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, selected["sla_violation_rate"], color="#b42318", marker="o", label="SLA Violation Rate")
    ax2.set_ylabel("SLA Violation Rate")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.suptitle("Cross-Region Delay Sensitivity")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spatial_migration_bar(results_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["algorithm"], results_df["remote_task_count"], color="#3b5bdb")
    ax.set_ylabel("Remote Task Count")
    ax.set_title("Spatial Migration Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_load_shift_curve(hourly_df: pd.DataFrame, metrics, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = hourly_df["hour"]
    ax.plot(hours, metrics.hourly_power_kw, marker="o", label="Local Power")
    ax.plot(hours, metrics.hourly_effective_power_kw, marker="s", label="Local + Remote Energy Equivalent")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power / Energy Equivalent (kW)")
    ax.set_title("Local Load Curve With Spatial Migration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_space_ablation(ablation_df: pd.DataFrame, output_path: str | Path) -> None:
    scenario_order = ["完整Proposed", "无空间迁移", "无时间迁移"]
    label_map = {
        "完整Proposed": "Time+Space",
        "无空间迁移": "Time Only",
        "无时间迁移": "Space Only",
    }
    selected = ablation_df[ablation_df["algorithm"].isin(scenario_order)].copy()
    if selected.empty:
        return
    selected["algorithm"] = pd.Categorical(selected["algorithm"], categories=scenario_order, ordered=True)
    selected = selected.sort_values("algorithm")
    labels = [label_map[str(value)] for value in selected["algorithm"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(labels, selected["total_cost"], color="#1864ab", alpha=0.85)
    ax1.set_ylabel("Total Cost")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(labels, selected["avg_delay_hours"], color="#d9480f", marker="o")
    ax2.set_ylabel("Average Delay (hours)")

    fig.suptitle("Time-Space Migration Ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_space_migration_effect(effect_df: pd.DataFrame, output_path: str | Path) -> None:
    if effect_df.empty:
        return

    x_labels = effect_df["scenario"].astype(str).tolist()
    x = list(range(len(effect_df)))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, effect_df["total_cost"], color="#1864ab", alpha=0.85, label="Total Cost")
    ax1.set_ylabel("Total Cost")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, effect_df["avg_delay_hours"], color="#d9480f", marker="o", label="Average Delay")
    ax2.set_ylabel("Average Delay (hours)")

    fig.suptitle("Time-Space Migration Effect")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_power_breakdown_stack(power_breakdown_df: pd.DataFrame, output_path: str | Path) -> None:
    required_cols = [
        "hour",
        "cpu_it_power_kw",
        "gpu_it_power_kw",
        "cooling_power_kw",
        "fixed_power_kw",
    ]
    if power_breakdown_df.empty or any(col not in power_breakdown_df.columns for col in required_cols):
        return

    hours = power_breakdown_df["hour"].tolist()
    bottom = [0.0 for _ in hours]
    stacks = [
        ("cpu_it_power_kw", "CPU IT Power"),
        ("gpu_it_power_kw", "GPU IT Power"),
        ("cooling_power_kw", "Cooling Power"),
        ("fixed_power_kw", "Fixed Power"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in stacks:
        values = power_breakdown_df[col].astype(float).tolist()
        ax.bar(hours, values, bottom=bottom, label=label)
        bottom = [base + value for base, value in zip(bottom, values)]

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(hours)
    ax.set_title("Hourly Power Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rolling_vs_static(rolling_df: pd.DataFrame, output_path: str | Path) -> None:
    selected = rolling_df[rolling_df["scope"] == "overall"].copy()
    if selected.empty:
        return

    algorithms = selected["algorithm"].astype(str).tolist()
    metrics = [
        ("total_cost", "Total Cost"),
        ("avg_delay_hours", "Average Delay (hours)"),
        ("sla_violation_rate", "SLA Violation Rate"),
        ("remote_task_count", "Remote Task Count"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(algorithms, selected[col])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=12)

    fig.suptitle("Rolling-Proposed vs Static-Proposed")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
