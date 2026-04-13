from pathlib import Path
import pandas as pd


def load_hourly_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"输入数据不存在: {path}")

    df = pd.read_csv(path)

    required_cols = {"hour", "arrival_rate", "price", "carbon_factor"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    if len(df) != 24:
        raise ValueError("当前版本要求输入 24 行小时数据。")

    df = df.sort_values("hour").reset_index(drop=True)
    return df