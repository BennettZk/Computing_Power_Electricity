from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.resource import ResourcePool, ResourceType
from models.task import generate_synthetic_tasks, load_tasks_csv


def load_yaml_like(path: str | Path) -> dict:
    """读取配置文件。当前配置使用 JSON 兼容写法，保留 .yaml 文件名便于论文说明。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_all_configs(config_dir: str | Path = "config") -> dict[str, dict]:
    """一次性加载基础参数、电价参数、资源参数和实验参数。"""
    config_dir = Path(config_dir)
    return {
        "base": load_yaml_like(config_dir / "base.yaml"),
        "price": load_yaml_like(config_dir / "price.yaml"),
        "resource": load_yaml_like(config_dir / "resource.yaml"),
        "experiment": load_yaml_like(config_dir / "experiment.yaml"),
    }


def load_resource_pool(resource_cfg: dict) -> ResourcePool:
    """根据 resource.yaml 构造 CPU/GPU 异构资源池对象。"""
    resources = {item["server_type"]: ResourceType(**item) for item in resource_cfg["resource_types"]}
    return ResourcePool(cpu=resources["cpu"], gpu=resources["gpu"])


def ensure_hourly_profile(base_cfg: dict, price_cfg: dict) -> pd.DataFrame:
    """保证小时级价格/负载数据存在；没有处理文件时从旧输入或默认数据生成。"""
    processed_path = Path(price_cfg["processed_profile_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists():
        return pd.read_csv(processed_path)

    legacy_path = Path(price_cfg["legacy_profile_path"])
    if legacy_path.exists():
        # 复用原论文复现代码中的 hourly_input.csv，避免推翻原数据流。
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
    """保证任务序列存在；首次运行时根据小时负载生成三类合成任务。"""
    tasks_path = Path(experiment_cfg["paths"]["tasks_path"])
    required_migration_cols = {"migratable", "migration_cost_weight", "migration_delay_penalty"}
    should_generate = not tasks_path.exists()
    if tasks_path.exists():
        existing_cols = set(pd.read_csv(tasks_path, nrows=1).columns)
        # 旧版合成任务没有空间迁移字段时，重新生成一次，确保 Proposed 能展示空间迁移能力。
        should_generate = not required_migration_cols.issubset(existing_cols)

    if should_generate:
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
    """加载实验所需的小时曲线和任务序列。"""
    hourly_df = ensure_hourly_profile(configs["base"], configs["price"])
    tasks_path = ensure_task_input(hourly_df, configs["base"], configs["experiment"])
    tasks = load_tasks_csv(tasks_path)
    return hourly_df, tasks


def ensure_output_dirs(experiment_cfg: dict) -> Path:
    """创建旧流程实验所需的输出目录集合。"""
    outputs_dir = Path(experiment_cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def write_summary(summary_path: str | Path, lines: list[str]) -> None:
    """把实验摘要文本写入指定文件。"""
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
