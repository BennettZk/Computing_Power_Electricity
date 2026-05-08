from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from utils.io_utils import ensure_dir, read_csv_required, save_csv, write_text
from utils.metrics import RESULT_CSV_COLUMN_MAPPING, export_csv_chinese, safe_divide
from utils.plotting import save_line_plot, save_multi_line_plot, setup_chinese_matplotlib


REGION_COLUMNS = {
    "Asia": "asia_export_tokens",
    "Europe": "europe_export_tokens",
    "America": "america_export_tokens",
}

STRATEGY_LABELS = {
    "No-Export": "不出口",
    "Power-Margin-Only": "仅功率裕度",
    "Price-Driven": "电价驱动",
    "Latency-Aware": "时延感知",
    "RH-TEO": "RH-TEO策略",
}

REGION_LABELS = {
    "Asia": "亚洲",
    "Europe": "欧洲",
    "America": "美洲",
}

PARAMETER_LABELS = {
    "base_token_sla_hours": "SLA阈值",
    "latency_scale": "跨时区时延放大系数",
    "america_price_multiplier": "美洲价格倍率",
    "window_size_hours": "滚动窗口长度",
    "token_per_margin_unit_factor": "Token产能系数",
}


def _plot_ready_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Exclude the zero-export baseline from main comparison figures."""
    return summary[summary["strategy"] != "No-Export"].copy()


def _plot_ready_region(region_results: pd.DataFrame) -> pd.DataFrame:
    """Exclude the zero-export baseline from main region-export figures."""
    return region_results[region_results["strategy"] != "No-Export"].copy()


def _token_capacity(fusion_df: pd.DataFrame, config: dict) -> pd.Series:
    return fusion_df["power_margin_norm"].clip(lower=0.0) * float(config["token_per_margin_unit"])


def _local_hour(hour: int, region_cfg: dict) -> int:
    return int((hour + int(region_cfg.get("timezone_offset", 0))) % 24)


def _demand_multiplier(hour: int, region_cfg: dict) -> float:
    local_hour = _local_hour(hour, region_cfg)
    return 1.2 if local_hour in set(region_cfg.get("demand_peak_hours", [])) else 0.9


def _electricity_price_per_kwh(price_value: float) -> float:
    # NBSDC cluster data is typically USD/MWh. If a small synthetic-style value is
    # supplied, treat it as already being a per-kWh tariff.
    return float(price_value) / 1000.0 if float(price_value) > 5.0 else float(price_value)


def _energy_per_million_tokens(margin_norm: float) -> float:
    margin_norm = max(0.0, min(1.0, float(margin_norm)))
    return 0.35 + 0.65 * (1.0 - margin_norm)


def _evaluate_hour(
    hour: int,
    row: pd.Series,
    allocation: dict[str, float],
    config: dict,
    previous_export_tokens: float,
) -> dict:
    regions = config["regions"]
    capacity = float(row["token_capacity"])
    export_tokens = float(sum(allocation.values()))
    export_million = export_tokens / 1_000_000.0
    utilization = safe_divide(export_tokens, capacity) if capacity > 0 else 0.0
    energy_per_million = _energy_per_million_tokens(float(row["power_margin_norm"]))
    electricity_price = _electricity_price_per_kwh(float(row["price"]))
    token_energy_cost = export_million * energy_per_million * electricity_price

    revenue = 0.0
    delay_weighted_sum = 0.0
    violation_tokens = 0.0
    for region, tokens in allocation.items():
        tokens = float(tokens)
        if tokens <= 0:
            continue
        region_cfg = regions[region]
        revenue += tokens / 1_000_000.0 * float(region_cfg["price_per_million_tokens"]) * _demand_multiplier(hour, region_cfg)
        delay = float(region_cfg["latency_hours"]) + 0.25 * utilization
        delay_weighted_sum += delay * tokens
        if delay > float(config["base_token_sla_hours"]):
            violation_tokens += tokens

    avg_delay = safe_divide(delay_weighted_sum, export_tokens)
    sla_violation_rate = safe_divide(violation_tokens, export_tokens)
    ramp_tokens = abs(export_tokens - float(previous_export_tokens))
    ramp_limit = float(config.get("ramp_limit_ratio", 0.35)) * float(config["token_per_margin_unit"])
    ramp_excess_million = max(0.0, ramp_tokens - ramp_limit) / 1_000_000.0

    utility = (
        revenue
        - float(config["energy_cost_weight"]) * token_energy_cost
        - float(config["latency_penalty_weight"]) * avg_delay * export_million
        - float(config["sla_penalty_weight"]) * float(config.get("sla_penalty_unit_scale", 0.05)) * sla_violation_rate * export_million
        - float(config["ramp_penalty_weight"]) * ramp_excess_million
    )

    result = {
        "hour": int(hour),
        "token_capacity": capacity,
        "export_tokens": export_tokens,
        "token_export_revenue": revenue,
        "token_energy_cost": token_energy_cost,
        "net_token_profit": revenue - token_energy_cost,
        "avg_cross_timezone_delay": avg_delay,
        "token_sla_violation_rate": sla_violation_rate,
        "energy_per_million_tokens": energy_per_million if export_tokens > 0 else 0.0,
        "cost_per_million_tokens": safe_divide(token_energy_cost, export_million),
        "utility": utility,
        "export_ramp_tokens": ramp_tokens,
    }
    for region, column in REGION_COLUMNS.items():
        result[column] = float(allocation.get(region, 0.0))
    return result


def _zero_alloc() -> dict[str, float]:
    return {region: 0.0 for region in REGION_COLUMNS}


def _shares_grid(step: float = 0.25) -> list[dict[str, float]]:
    values = np.arange(0.0, 1.0 + step / 2, step)
    shares: list[dict[str, float]] = []
    for asia, europe in product(values, values):
        america = 1.0 - float(asia) - float(europe)
        if america < -1e-9:
            continue
        shares.append({"Asia": float(asia), "Europe": float(europe), "America": max(0.0, float(america))})
    return shares


def _allocation_from_share(capacity: float, total_ratio: float, share: dict[str, float]) -> dict[str, float]:
    total = max(0.0, min(1.0, float(total_ratio))) * max(0.0, float(capacity))
    return {region: total * share.get(region, 0.0) for region in REGION_COLUMNS}


def _run_fixed_strategy(strategy: str, hourly: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    previous_export = 0.0
    price_median = hourly["price"].median()
    margin_q75 = hourly["power_margin_norm"].quantile(0.75)
    highest_price_region = max(config["regions"].items(), key=lambda item: item[1]["price_per_million_tokens"])[0]

    for _, row in hourly.iterrows():
        hour = int(row["hour"])
        capacity = float(row["token_capacity"])
        if strategy == "No-Export" or capacity <= 0:
            allocation = _zero_alloc()
        elif strategy == "Power-Margin-Only":
            allocation = {region: capacity * 0.60 / len(REGION_COLUMNS) for region in REGION_COLUMNS}
        elif strategy == "Price-Driven":
            should_export = float(row["price"]) <= price_median or float(row["power_margin_norm"]) >= margin_q75
            allocation = _zero_alloc()
            if should_export:
                allocation[highest_price_region] = capacity * 0.75
        elif strategy == "Latency-Aware":
            allocation = {"Asia": capacity * 0.55 * 0.80, "Europe": capacity * 0.55 * 0.15, "America": capacity * 0.55 * 0.05}
        else:
            raise ValueError(f"Unknown fixed strategy: {strategy}")
        evaluated = _evaluate_hour(hour, row, allocation, config, previous_export)
        evaluated["strategy"] = strategy
        rows.append(evaluated)
        previous_export = evaluated["export_tokens"]
    return pd.DataFrame(rows)


def _run_rh_teo(hourly: pd.DataFrame, config: dict) -> pd.DataFrame:
    shares = _shares_grid(step=0.25)
    total_ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    window_size = int(config["window_size_hours"])
    rows: list[dict] = []
    previous_export = 0.0

    for current_idx in range(len(hourly)):
        best_score = -float("inf")
        best_allocation = _zero_alloc()
        window = hourly.iloc[current_idx : min(len(hourly), current_idx + window_size)]
        for total_ratio in total_ratios:
            for share in shares:
                candidate_previous = previous_export
                candidate_score = 0.0
                first_allocation = None
                for _, future_row in window.iterrows():
                    allocation = _allocation_from_share(float(future_row["token_capacity"]), total_ratio, share)
                    evaluated = _evaluate_hour(int(future_row["hour"]), future_row, allocation, config, candidate_previous)
                    candidate_score += float(evaluated["utility"])
                    candidate_previous = float(evaluated["export_tokens"])
                    if first_allocation is None:
                        first_allocation = allocation
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_allocation = first_allocation or _zero_alloc()

        current_row = hourly.iloc[current_idx]
        evaluated = _evaluate_hour(int(current_row["hour"]), current_row, best_allocation, config, previous_export)
        evaluated["strategy"] = "RH-TEO"
        rows.append(evaluated)
        previous_export = evaluated["export_tokens"]
    return pd.DataFrame(rows)


def _summarize_strategy(hourly_result: pd.DataFrame) -> dict:
    total_export = float(hourly_result["export_tokens"].sum())
    total_million = total_export / 1_000_000.0
    revenue = float(hourly_result["token_export_revenue"].sum())
    energy_cost = float(hourly_result["token_energy_cost"].sum())
    delay = safe_divide((hourly_result["avg_cross_timezone_delay"] * hourly_result["export_tokens"]).sum(), total_export)
    sla = safe_divide((hourly_result["token_sla_violation_rate"] * hourly_result["export_tokens"]).sum(), total_export)
    energy = safe_divide((hourly_result["energy_per_million_tokens"] * hourly_result["export_tokens"]).sum(), total_export)
    return {
        "strategy": hourly_result["strategy"].iloc[0],
        "total_export_tokens": total_export,
        "token_export_revenue": revenue,
        "token_energy_cost": energy_cost,
        "net_token_profit": revenue - energy_cost,
        "avg_cross_timezone_delay": delay,
        "token_sla_violation_rate": sla,
        "energy_per_million_tokens": energy,
        "cost_per_million_tokens": safe_divide(energy_cost, total_million),
        "asia_export_tokens": float(hourly_result["asia_export_tokens"].sum()),
        "europe_export_tokens": float(hourly_result["europe_export_tokens"].sum()),
        "america_export_tokens": float(hourly_result["america_export_tokens"].sum()),
    }


def _run_all_strategies(hourly: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy_frames = [
        _run_fixed_strategy("No-Export", hourly, config),
        _run_fixed_strategy("Power-Margin-Only", hourly, config),
        _run_fixed_strategy("Price-Driven", hourly, config),
        _run_fixed_strategy("Latency-Aware", hourly, config),
        _run_rh_teo(hourly, config),
    ]
    hourly_results = pd.concat(strategy_frames, ignore_index=True)
    summary = pd.DataFrame([_summarize_strategy(frame) for frame in strategy_frames])

    region_rows: list[dict] = []
    for _, row in summary.iterrows():
        total = float(row["total_export_tokens"])
        for region, column in REGION_COLUMNS.items():
            tokens = float(row[column])
            region_rows.append({"strategy": row["strategy"], "region": region, "export_tokens": tokens, "share": safe_divide(tokens, total)})
    region_results = pd.DataFrame(region_rows)
    return summary, hourly_results, region_results


def _run_sensitivity(hourly: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []

    def collect(parameter: str, value, modified_config: dict) -> None:
        rh = _run_rh_teo(hourly.assign(token_capacity=_token_capacity(hourly, modified_config)), modified_config)
        summary = _summarize_strategy(rh)
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "total_export_tokens": summary["total_export_tokens"],
                "net_token_profit": summary["net_token_profit"],
                "avg_cross_timezone_delay": summary["avg_cross_timezone_delay"],
                "token_sla_violation_rate": summary["token_sla_violation_rate"],
                "asia_export_tokens": summary["asia_export_tokens"],
                "europe_export_tokens": summary["europe_export_tokens"],
                "america_export_tokens": summary["america_export_tokens"],
                "america_export_share": safe_divide(summary["america_export_tokens"], summary["total_export_tokens"]),
            }
        )

    for sla_hours in [0.12, 0.20, 0.30, 0.40]:
        cfg = deepcopy(config)
        cfg["base_token_sla_hours"] = sla_hours
        collect("base_token_sla_hours", sla_hours, cfg)

    for scale in [0.8, 1.0, 1.5, 2.0]:
        cfg = deepcopy(config)
        for region_cfg in cfg["regions"].values():
            region_cfg["latency_hours"] = float(region_cfg["latency_hours"]) * scale
        collect("latency_scale", scale, cfg)

    base_america_price = float(config["regions"]["America"]["price_per_million_tokens"])
    for multiplier in [1.0, 1.2, 1.5, 1.8]:
        cfg = deepcopy(config)
        cfg["regions"]["America"]["price_per_million_tokens"] = base_america_price * multiplier
        collect("america_price_multiplier", multiplier, cfg)

    for window in [2, 4, 6]:
        cfg = deepcopy(config)
        cfg["window_size_hours"] = window
        collect("window_size_hours", window, cfg)

    base_token_unit = float(config["token_per_margin_unit"])
    for factor in [0.8, 1.0, 1.2]:
        cfg = deepcopy(config)
        cfg["token_per_margin_unit"] = base_token_unit * factor
        collect("token_per_margin_unit_factor", factor, cfg)

    return pd.DataFrame(rows)


def _plot_region_export(region_results: pd.DataFrame, output_dir: Path) -> None:
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    plot_df = region_results.copy()
    plot_df["strategy_label"] = plot_df["strategy"].map(STRATEGY_LABELS).fillna(plot_df["strategy"])
    pivot = plot_df.pivot(index="strategy_label", columns="region", values="export_tokens").fillna(0.0) / 1_000_000.0
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot))
    colors = {"Asia": "#2563eb", "Europe": "#16a34a", "America": "#dc2626"}
    for region in ["Asia", "Europe", "America"]:
        values = pivot[region].values if region in pivot.columns else np.zeros(len(pivot))
        ax.bar(pivot.index, values, bottom=bottom, label=REGION_LABELS[region], color=colors[region])
        bottom += values
    ax.set_title("不同地区Token出口量对比")
    ax.set_xlabel("策略")
    ax.set_ylabel("出口Token量/百万")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "token_export_by_region.png")
    plt.close(fig)


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    x_col = x
    if x == "strategy" and "strategy" in plot_df.columns:
        plot_df["strategy_label"] = plot_df["strategy"].map(STRATEGY_LABELS).fillna(plot_df["strategy"])
        x_col = "strategy_label"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(plot_df[x_col], plot_df[y], color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("策略")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sla = sensitivity[sensitivity["parameter"] == "base_token_sla_hours"]
    latency = sensitivity[sensitivity["parameter"] == "latency_scale"]
    america = sensitivity[sensitivity["parameter"] == "america_price_multiplier"]

    ax0b = axes[0].twinx()
    axes[0].plot(sla["value"], sla["token_sla_violation_rate"], marker="o", color="#dc2626", label="SLA违约率")
    ax0b.plot(sla["value"], sla["net_token_profit"], marker="s", color="#2563eb", label="净收益")
    axes[0].set_title("SLA阈值敏感性")
    axes[0].set_xlabel("SLA阈值/h")
    axes[0].set_ylabel("SLA违约率")
    ax0b.set_ylabel("净收益")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    ax1b = axes[1].twinx()
    axes[1].plot(latency["value"], latency["avg_cross_timezone_delay"], marker="o", color="#16a34a", label="平均时延")
    ax1b.plot(latency["value"], latency["net_token_profit"], marker="s", color="#2563eb", label="净收益")
    axes[1].set_title("跨时区时延放大敏感性")
    axes[1].set_xlabel("时延放大系数")
    axes[1].set_ylabel("平均时延/h")
    ax1b.set_ylabel("净收益")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    ax2b = axes[2].twinx()
    axes[2].plot(america["value"], america["america_export_share"], marker="o", color="#f97316", label="美洲占比")
    ax2b.plot(america["value"], america["net_token_profit"], marker="s", color="#2563eb", label="净收益")
    axes[2].set_title("美洲价格倍率敏感性")
    axes[2].set_xlabel("美洲价格倍率")
    axes[2].set_ylabel("美洲出口占比")
    ax2b.set_ylabel("净收益")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Token出口关键参数敏感性分析")
    fig.tight_layout()
    fig.savefig(output_dir / "token_sensitivity.png")
    plt.close(fig)


def _plot_outputs(summary: pd.DataFrame, hourly_results: pd.DataFrame, region_results: pd.DataFrame, sensitivity: pd.DataFrame, output_dir: Path) -> None:
    plot_summary = _plot_ready_summary(summary)
    plot_region_results = _plot_ready_region(region_results)
    capacity_df = hourly_results[hourly_results["strategy"] == "RH-TEO"][["hour", "token_capacity"]].copy()
    save_line_plot(
        capacity_df,
        x="hour",
        y="token_capacity",
        path=output_dir / "hourly_token_capacity.png",
        title="小时级Token产出能力",
        xlabel="小时",
        ylabel="Token产出能力",
    )
    _plot_region_export(plot_region_results, output_dir)
    _plot_bar(plot_summary, "strategy", "net_token_profit", output_dir / "token_profit_comparison.png", "不同Token出口策略净收益对比", "净收益")
    _plot_bar(plot_summary, "strategy", "avg_cross_timezone_delay", output_dir / "cross_timezone_latency.png", "不同Token出口策略跨时区服务时延对比", "平均时延/h")

    # Rebuild the curve from RH-TEO rows so the x-axis is the fused margin.
    rh_rows = hourly_results[hourly_results["strategy"] == "RH-TEO"].sort_values("token_capacity")
    margin_curve = rh_rows[["hour", "token_capacity"]].copy()
    if "power_margin_norm" in rh_rows.columns:
        margin_curve["power_margin_norm"] = rh_rows["power_margin_norm"]
    else:
        margin_curve["power_margin_norm"] = safe_divide(margin_curve["token_capacity"], margin_curve["token_capacity"].max())
    setup_chinese_matplotlib()
    import matplotlib.pyplot as plt

    rh_time = hourly_results[hourly_results["strategy"] == "RH-TEO"].sort_values("hour")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(rh_time["hour"], rh_time["power_margin_norm"], marker="o", linewidth=2, color="#2563eb", label="等效功率裕度")
    ax1.set_xlabel("小时")
    ax1.set_ylabel("等效功率裕度")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(rh_time["hour"], rh_time["token_capacity"], marker="s", linewidth=2, color="#dc2626", label="Token产出能力")
    ax2.set_ylabel("Token产出能力")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("等效功率裕度与Token产出能力时序关系")
    fig.tight_layout()
    fig.savefig(output_dir / "power_margin_token_capacity_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(margin_curve["power_margin_norm"], margin_curve["token_capacity"], marker="o", linewidth=2)
    ax.set_title("等效功率裕度到Token产出的转换关系")
    ax.set_xlabel("等效功率裕度")
    ax.set_ylabel("Token产出能力")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "power_to_token_curve.png")
    plt.close(fig)

    rh_alloc = hourly_results[hourly_results["strategy"] == "RH-TEO"][
        ["hour", "asia_export_tokens", "europe_export_tokens", "america_export_tokens"]
    ].copy()
    for col in ["asia_export_tokens", "europe_export_tokens", "america_export_tokens"]:
        rh_alloc[col] = rh_alloc[col] / 1_000_000.0
    save_multi_line_plot(
        rh_alloc,
        x="hour",
        y_columns=["asia_export_tokens", "europe_export_tokens", "america_export_tokens"],
        path=output_dir / "rh_teo_allocation_curve.png",
        title="RH-TEO策略下跨时区Token分配曲线",
        xlabel="小时",
        ylabel="出口Token量/百万",
        labels={"asia_export_tokens": "亚洲", "europe_export_tokens": "欧洲", "america_export_tokens": "美洲"},
    )
    _plot_sensitivity(sensitivity, output_dir)


def run_token_export(
    fusion_path: str | Path = "outputs/nbsdc_fusion/aligned_hourly_fusion.csv",
    config: dict | None = None,
    output_dir: str | Path = "outputs/token_export",
) -> dict[str, Path]:
    if config is None:
        raise ValueError("config is required")
    output_dir = ensure_dir(output_dir)
    fusion = read_csv_required(fusion_path, ["hour", "price", "power_margin_norm"])
    hourly = fusion[["hour", "price", "power_margin_norm"]].copy()
    hourly["token_capacity"] = _token_capacity(hourly, config)

    summary, hourly_results, region_results = _run_all_strategies(hourly, config)
    hourly_results = hourly_results.merge(hourly[["hour", "price", "power_margin_norm"]], on="hour", how="left")
    sensitivity = _run_sensitivity(hourly, config)

    results_path = save_csv(summary, output_dir / "token_export_results.csv")
    summary_cn = summary.copy()
    summary_cn["strategy"] = summary_cn["strategy"].map(STRATEGY_LABELS).fillna(summary_cn["strategy"])
    export_csv_chinese(summary_cn, output_dir / "token_export_results_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    hourly_path = save_csv(hourly_results, output_dir / "hourly_token_export.csv")
    region_path = save_csv(region_results, output_dir / "region_token_export.csv")
    sensitivity_path = save_csv(sensitivity, output_dir / "token_sensitivity_results.csv")
    sensitivity_cn = sensitivity.copy()
    sensitivity_cn["parameter"] = sensitivity_cn["parameter"].map(PARAMETER_LABELS).fillna(sensitivity_cn["parameter"])
    export_csv_chinese(sensitivity_cn, output_dir / "token_sensitivity_results_cn.csv", RESULT_CSV_COLUMN_MAPPING)
    _plot_outputs(summary, hourly_results, region_results, sensitivity, output_dir)

    best = summary.sort_values("net_token_profit", ascending=False).iloc[0]
    best_strategy_label = STRATEGY_LABELS.get(str(best["strategy"]), str(best["strategy"]))
    rh_row = summary.loc[summary["strategy"] == "RH-TEO"].iloc[0]
    summary_lines = [
        "Token出口优化实验摘要",
        "=" * 28,
        "本实验基于 NBSDC 三层融合得到的等效功率裕度构建 Token 出口扩展场景。",
        "NBSDC 源数据不被解释为真实 Token 请求记录。",
        "",
        f"Token产能系数: {config['token_per_margin_unit']}",
        f"滚动窗口长度: {config['window_size_hours']} 小时",
        f"最优策略: {best_strategy_label}",
        f"最优策略净收益: {best['net_token_profit']:.6f}",
        f"RH-TEO总出口Token量: {rh_row['total_export_tokens']:.2f}",
        f"RH-TEO SLA违约率: {rh_row['token_sla_violation_rate']:.6f}",
        "",
        "“不出口”作为零出口基准保留在结果表中，不参与主要图表展示；主要图表比较的是实际发生 Token 出口的策略。",
        "“不出口”无跨时区服务时延，因此不参与跨时区服务时延图比较。",
        "",
        "生成图表: hourly_token_capacity, token_export_by_region, token_profit_comparison, cross_timezone_latency, power_margin_token_capacity_timeseries, power_to_token_curve, rh_teo_allocation_curve, token_sensitivity。",
    ]
    summary_path = write_text(output_dir / "token_export_summary.txt", summary_lines)

    return {
        "token_export_results": results_path,
        "hourly_token_export": hourly_path,
        "region_token_export": region_path,
        "token_sensitivity_results": sensitivity_path,
        "token_sensitivity_results_cn": output_dir / "token_sensitivity_results_cn.csv",
        "summary": summary_path,
    }
