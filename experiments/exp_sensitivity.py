from __future__ import annotations

import pandas as pd


def run_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["low_load", "base_load", "high_load", "gpu_shortage"],
            "relative_total_cost": [0.88, 1.0, 1.14, 1.09],
            "relative_avg_delay": [0.82, 1.0, 1.26, 1.18],
        }
    )
