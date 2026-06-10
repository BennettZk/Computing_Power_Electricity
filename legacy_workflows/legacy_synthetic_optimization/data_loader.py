from pathlib import Path

import pandas as pd

from utils.io_utils import ensure_hourly_profile, load_all_configs


def load_hourly_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """加载小时级电价、负载和功率输入数据。"""
    path = Path(csv_path) if csv_path is not None else None
    if path is not None and path.exists():
        df = pd.read_csv(path)
    else:
        configs = load_all_configs()
        df = ensure_hourly_profile(configs["base"], configs["price"])

    required_cols = {"hour", "arrival_rate", "price", "carbon_factor"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Hourly input is missing columns: {sorted(missing)}")

    if len(df) != 24:
        raise ValueError("The current implementation expects 24 hourly rows.")

    return df.sort_values("hour").reset_index(drop=True)
