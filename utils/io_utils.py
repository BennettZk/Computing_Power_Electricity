from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.resource import ResourcePool, ResourceType
from models.task import generate_synthetic_tasks, load_tasks_csv


def load_yaml_like(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_all_configs(config_dir: str | Path = "config") -> dict[str, dict]:
    config_dir = Path(config_dir)
    return {
        "base": load_yaml_like(config_dir / "base.yaml"),
        "price": load_yaml_like(config_dir / "price.yaml"),
        "resource": load_yaml_like(config_dir / "resource.yaml"),
        "experiment": load_yaml_like(config_dir / "experiment.yaml"),
    }


def load_resource_pool(resource_cfg: dict) -> ResourcePool:
    resources = {item["server_type"]: ResourceType(**item) for item in resource_cfg["resource_types"]}
    return ResourcePool(cpu=resources["cpu"], gpu=resources["gpu"])


def ensure_hourly_profile(base_cfg: dict, price_cfg: dict) -> pd.DataFrame:
    processed_path = Path(price_cfg["processed_profile_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists():
        return pd.read_csv(processed_path)

    legacy_path = Path(price_cfg["legacy_profile_path"])
    if legacy_path.exists():
        legacy_df = pd.read_csv(legacy_path).sort_values("hour").reset_index(drop=True)
    else:
        legacy_df = pd.DataFrame(
            {
                "hour": list(range(base_cfg["time"]["hours"])),
                "arrival_rate": [120, 110, 105, 95, 90, 95, 125, 155, 190, 210, 225, 235, 220, 215, 205, 210, 235, 260, 280, 270, 230, 190, 150, 130],
                "price": [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.88, 0.88, 1.18, 1.18, 1.18, 1.18, 0.88, 0.88, 0.88, 0.88, 1.18, 1.18, 1.18, 1.18, 0.88, 0.88, 0.32, 0.32],
                "carbon_factor": [0.52, 0.51, 0.50, 0.49, 0.49, 0.50, 0.53, 0.56, 0.61, 0.64, 0.66, 0.67, 0.65, 0.63, 0.62, 0.63, 0.66, 0.69, 0.71, 0.70, 0.64, 0.60, 0.56, 0.54],
            }
        )

    hourly_df = legacy_df.copy()
    hourly_df["renewable_ratio"] = [0.36, 0.38, 0.40, 0.42, 0.45, 0.43, 0.30, 0.28, 0.24, 0.22, 0.21, 0.20, 0.26, 0.28, 0.30, 0.27, 0.22, 0.20, 0.18, 0.19, 0.26, 0.30, 0.34, 0.35]
    hourly_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    return hourly_df


def ensure_task_input(hourly_df: pd.DataFrame, base_cfg: dict, experiment_cfg: dict) -> Path:
    tasks_path = Path(experiment_cfg["paths"]["tasks_path"])
    if not tasks_path.exists():
        constraints = base_cfg["constraints"]
        generate_synthetic_tasks(
            hourly_df=hourly_df,
            output_path=tasks_path,
            seed=int(experiment_cfg["seed"]),
            max_delay_slots_tolerant=int(constraints["max_delay_slots_tolerant"]),
            max_delay_slots_token=int(constraints["max_delay_slots_token"]),
        )
    return tasks_path


def ensure_inputs(configs: dict[str, dict]) -> tuple[pd.DataFrame, list]:
    hourly_df = ensure_hourly_profile(configs["base"], configs["price"])
    tasks_path = ensure_task_input(hourly_df, configs["base"], configs["experiment"])
    tasks = load_tasks_csv(tasks_path)
    return hourly_df, tasks


def ensure_output_dirs(experiment_cfg: dict) -> Path:
    outputs_dir = Path(experiment_cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def write_summary(summary_path: str | Path, lines: list[str]) -> None:
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
