from __future__ import annotations

from pathlib import Path

from experiments.exp_nbsdc_fusion import run_nbsdc_fusion


REQUIRED_FILES = [
    "cluster_power_price_5min.csv",
    "server_tasks_5min_raw_mapped.csv",
    "server_tasks_24h.csv",
    "hourly_input_24h.csv",
    "chip_dvfs.csv",
]


def main() -> None:
    data_dir = Path("data/real_case")
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        print("Missing cleaned NBSDC CSV files:")
        for name in missing:
            print(f"- {data_dir / name}")
        print("Please run: python scripts/prepare_uploaded_case_dataset.py")
        raise SystemExit(1)

    outputs = run_nbsdc_fusion(data_dir=data_dir)
    print("NBSDC fusion finished.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
