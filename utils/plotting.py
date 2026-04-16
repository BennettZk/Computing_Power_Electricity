from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    ax.bar(results_df["algorithm"], results_df["total_energy_kwh"], color=["#567d46", "#c98d00", "#7f8c8d", "#0b7285"])
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


def plot_ablation_results(ablation_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_labels = ["Full", "No Price", "No Priority", "No Hetero"][: len(ablation_df)]
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
    x_labels = ["Base", "Load 0.8x", "Load 1.2x", "GPU Half", "Power Tight"][: len(sensitivity_df)]
    ax1.plot(x_labels, sensitivity_df["total_cost"], marker="o", color="#0b7285", label="Total Cost")
    ax1.set_ylabel("Total Cost")
    ax1.tick_params(axis="x", rotation=18)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, sensitivity_df["sla_violation_rate"], marker="s", color="#b42318", label="SLA Violation")
    ax2.set_ylabel("SLA Violation Rate")

    fig.suptitle("Sensitivity Study: Cost and SLA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
