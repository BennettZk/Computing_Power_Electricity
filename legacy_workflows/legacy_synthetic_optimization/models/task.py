from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import pandas as pd


def _to_bool(value) -> bool:
    """兼容 CSV 中的 True/False、1/0、yes/no 等布尔写法。"""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


@dataclass
class Task:
    """任务到达序列中的单个任务，保留论文实验所需的资源需求和截止时间字段。"""

    task_id: str
    arrival_time: int
    task_type: str
    cpu_demand: float
    gpu_demand: float
    memory_demand: float
    bandwidth_demand: float
    token_amount: float
    deadline: int
    priority: int
    migratable: bool = False
    migration_cost_weight: float = 1.0
    migration_delay_penalty: float = 0.0

    @property
    def delay_tolerant(self) -> bool:
        """时延容忍任务和 token 批任务允许在高电价时段被延后处理。"""
        return self.task_type in {"delay_tolerant", "token_batch"}

    @property
    def preferred_resource(self) -> str:
        """GPU 需求大于 0 或 token 批任务优先放入 GPU 队列，其余进入 CPU 队列。"""
        return "gpu" if self.gpu_demand > 0 or self.task_type == "token_batch" else "cpu"

    @property
    def service_demand(self) -> float:
        """调度时只取任务首选资源上的计算需求作为队列服务量近似。"""
        return self.gpu_demand if self.preferred_resource == "gpu" else self.cpu_demand

    @property
    def deadline_slot(self) -> int:
        return int(self.deadline)


def load_tasks_csv(csv_path: str | Path) -> list[Task]:
    """从 CSV 读取任务序列，并校验必需字段是否完整。"""
    df = pd.read_csv(csv_path)
    required_cols = {
        "task_id",
        "arrival_time",
        "task_type",
        "cpu_demand",
        "gpu_demand",
        "memory_demand",
        "bandwidth_demand",
        "token_amount",
        "deadline",
        "priority",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Task input is missing columns: {sorted(missing)}")

    # 兼容旧版任务数据：没有空间迁移字段时使用默认值。
    if "migratable" not in df.columns:
        df["migratable"] = False
    if "migration_cost_weight" not in df.columns:
        df["migration_cost_weight"] = 1.0
    if "migration_delay_penalty" not in df.columns:
        df["migration_delay_penalty"] = 0.0

    tasks = [
        Task(
            task_id=str(row.task_id),
            arrival_time=int(row.arrival_time),
            task_type=str(row.task_type),
            cpu_demand=float(row.cpu_demand),
            gpu_demand=float(row.gpu_demand),
            memory_demand=float(row.memory_demand),
            bandwidth_demand=float(row.bandwidth_demand),
            token_amount=float(row.token_amount),
            deadline=int(row.deadline),
            priority=int(row.priority),
            migratable=_to_bool(row.migratable),
            migration_cost_weight=float(row.migration_cost_weight),
            migration_delay_penalty=float(row.migration_delay_penalty),
        )
        for row in df.itertuples(index=False)
    ]
    return tasks


def tasks_to_dataframe(tasks: list[Task]) -> pd.DataFrame:
    return pd.DataFrame([asdict(task) for task in tasks])


def generate_synthetic_tasks(
    hourly_df: pd.DataFrame,
    output_path: str | Path,
    seed: int,
    max_delay_slots_tolerant: int,
    max_delay_slots_token: int,
) -> pd.DataFrame:
    """根据小时级负载生成可复现实验用的三类合成任务。"""
    rng = random.Random(seed)
    tasks: list[Task] = []

    for row in hourly_df.itertuples(index=False):
        hour = int(row.hour)
        total_arrival = int(row.arrival_rate)
        sensitive_count = int(round(total_arrival * 0.50))
        tolerant_count = int(round(total_arrival * 0.28))
        token_count = max(total_arrival - sensitive_count - tolerant_count, 0)

        # 时延敏感任务要求尽快完成，主要消耗 CPU 资源。
        for idx in range(sensitive_count):
            migratable = rng.random() < 0.03
            tasks.append(
                Task(
                    task_id=f"s-{hour:02d}-{idx:03d}",
                    arrival_time=hour,
                    task_type="delay_sensitive",
                    cpu_demand=round(rng.uniform(0.9, 1.4), 3),
                    gpu_demand=0.0,
                    memory_demand=round(rng.uniform(4.0, 10.0), 3),
                    bandwidth_demand=round(rng.uniform(0.4, 1.0), 3),
                    token_amount=0.0,
                    deadline=min(hour + 1, 23),
                    priority=3,
                    migratable=migratable,
                    migration_cost_weight=1.5,
                    migration_delay_penalty=0.18 if migratable else 0.0,
                )
            )

        # 时延容忍任务可被需求响应策略延后到低电价时段。
        for idx in range(tolerant_count):
            slack = rng.randint(2, max(2, max_delay_slots_tolerant))
            migratable = rng.random() < 0.62
            tasks.append(
                Task(
                    task_id=f"t-{hour:02d}-{idx:03d}",
                    arrival_time=hour,
                    task_type="delay_tolerant",
                    cpu_demand=round(rng.uniform(0.8, 1.2), 3),
                    gpu_demand=0.0,
                    memory_demand=round(rng.uniform(3.0, 9.0), 3),
                    bandwidth_demand=round(rng.uniform(0.3, 0.9), 3),
                    token_amount=round(rng.uniform(120.0, 480.0), 3),
                    deadline=min(hour + slack, 23),
                    priority=2,
                    migratable=migratable,
                    migration_cost_weight=0.9,
                    migration_delay_penalty=0.06 if migratable else 0.0,
                )
            )

        # token_batch 用于轻量表示 AI 推理/训练类任务，不涉及链上交易。
        for idx in range(token_count):
            slack = rng.randint(2, max(2, max_delay_slots_token))
            tokens = rng.randint(1200, 4800)
            tasks.append(
                Task(
                    task_id=f"g-{hour:02d}-{idx:03d}",
                    arrival_time=hour,
                    task_type="token_batch",
                    cpu_demand=round(rng.uniform(0.2, 0.6), 3),
                    gpu_demand=round(rng.uniform(0.8, 1.4), 3),
                    memory_demand=round(rng.uniform(8.0, 18.0), 3),
                    bandwidth_demand=round(rng.uniform(0.8, 1.4), 3),
                    token_amount=float(tokens),
                    deadline=min(hour + slack, 23),
                    priority=1,
                    migratable=True,
                    migration_cost_weight=0.95,
                    migration_delay_penalty=0.06,
                )
            )

    output_df = tasks_to_dataframe(tasks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_df


def aggregate_tasks_by_slot(tasks: list[Task], hours: int) -> pd.DataFrame:
    """按时隙聚合任务数量和资源需求，供基线算法估算开机台数。"""
    rows: list[dict[str, float]] = []
    for hour in range(hours):
        slot_tasks = [task for task in tasks if task.arrival_time == hour]
        cpu_tasks = [task for task in slot_tasks if task.preferred_resource == "cpu"]
        gpu_tasks = [task for task in slot_tasks if task.preferred_resource == "gpu"]
        migratable_tasks = [task for task in slot_tasks if task.migratable]
        rows.append(
            {
                "hour": hour,
                "task_count": len(slot_tasks),
                "cpu_task_count": len(cpu_tasks),
                "gpu_task_count": len(gpu_tasks),
                "delay_sensitive_count": sum(task.task_type == "delay_sensitive" for task in slot_tasks),
                "delay_tolerant_count": sum(task.task_type == "delay_tolerant" for task in slot_tasks),
                "token_batch_count": sum(task.task_type == "token_batch" for task in slot_tasks),
                "cpu_compute_demand": sum(task.cpu_demand for task in cpu_tasks),
                "gpu_compute_demand": sum(task.gpu_demand for task in gpu_tasks),
                "memory_demand": sum(task.memory_demand for task in slot_tasks),
                "bandwidth_demand": sum(task.bandwidth_demand for task in slot_tasks),
                "token_amount": sum(task.token_amount for task in slot_tasks),
                "migratable_task_count": len(migratable_tasks),
                "migratable_cpu_demand": sum(task.cpu_demand for task in migratable_tasks),
                "migratable_gpu_demand": sum(task.gpu_demand for task in migratable_tasks),
            }
        )
    return pd.DataFrame(rows)
