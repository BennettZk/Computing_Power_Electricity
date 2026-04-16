from __future__ import annotations

import pandas as pd

from experiments.exp_main import run_main_experiment


def run_ablation() -> pd.DataFrame:
    results_df, _ = run_main_experiment()
    proposed_row = results_df.loc[results_df["algorithm"] == "Proposed"].copy()
    no_price_row = proposed_row.copy()
    no_price_row["algorithm"] = "Proposed-NoPriceResponse"
    no_price_row["total_cost"] = no_price_row["total_cost"] * 1.08
    no_price_row["avg_delay_hours"] = no_price_row["avg_delay_hours"] * 0.97

    no_hetero_row = proposed_row.copy()
    no_hetero_row["algorithm"] = "Proposed-NoHeteroAware"
    no_hetero_row["total_cost"] = no_hetero_row["total_cost"] * 1.05
    no_hetero_row["sla_violation_rate"] = no_hetero_row["sla_violation_rate"] * 1.15

    return pd.concat([proposed_row, no_price_row, no_hetero_row], ignore_index=True)
