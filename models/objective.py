from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from models.delay_model import estimate_delay_statistics
from models.power_model import hourly_heterogeneous_power_kw, total_carbon_emission, total_energy_cost
from models.resource import ResourcePool
from models.task import Task
from utils.metrics import compute_load_imbalance, compute_peak_valley_gap, safe_divide


@dataclass(frozen=True)
class SchedulePlan:
    """优化器输出的调度方案：每小时 CPU/GPU 开机台数与可延迟任务比例。"""

    cpu_servers: np.ndarray
    gpu_servers: np.ndarray
    defer_ratio: np.ndarray
    migration_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


@dataclass
class TaskState:
    """仿真过程中使用的任务状态，记录任务是否已经被延迟到未来时隙。"""

    task_id: str
    arrival_slot: int
    task_type: str
    priority: int
    deadline_slot: int
    preferred_resource: str
    service_demand: float
    memory_demand: float
    bandwidth_demand: float
    token_amount: float
    delay_tolerant: bool
    earliest_service_slot: int
    migratable: bool
    migration_cost_weight: float
    migration_delay_penalty: float
    executed_remotely: bool = False


@dataclass
class SimulationResult:
    """一次调度仿真的完整评价指标，用于结果表、目标函数和绘图。"""

    total_energy_kwh: float
    total_cost: float
    total_carbon: float
    avg_delay_hours: float
    completion_rate: float
    sla_violation_rate: float
    delay_sensitive_violation_rate: float
    avg_cpu_utilization: float
    avg_gpu_utilization: float
    load_imbalance: float
    peak_valley_gap_kw: float
    total_tokens: float
    unit_token_energy_kwh_per_million: float
    unit_token_cost_per_million: float
    power_limit_violation_hours: int
    peak_limit_violation_hours: int
    remote_task_count: int
    remote_completion_rate: float
    remote_cost: float
    remote_energy_kwh: float
    migration_delay_hours: float
    total_completed_tasks: int
    total_tasks: int
    objective_penalty: float
    hourly_power_kw: list[float] = field(default_factory=list)
    hourly_delay_hours: list[float] = field(default_factory=list)
    hourly_completion_rate: list[float] = field(default_factory=list)
    hourly_cpu_utilization: list[float] = field(default_factory=list)
    hourly_gpu_utilization: list[float] = field(default_factory=list)
    hourly_cpu_servers: list[int] = field(default_factory=list)
    hourly_gpu_servers: list[int] = field(default_factory=list)
    hourly_remote_energy_kwh: list[float] = field(default_factory=list)
    hourly_remote_task_count: list[int] = field(default_factory=list)
    hourly_effective_power_kw: list[float] = field(default_factory=list)
    deferred_task_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_energy_kwh": self.total_energy_kwh,
            "total_cost": self.total_cost,
            "total_carbon": self.total_carbon,
            "avg_delay_hours": self.avg_delay_hours,
            "completion_rate": self.completion_rate,
            "sla_violation_rate": self.sla_violation_rate,
            "delay_sensitive_violation_rate": self.delay_sensitive_violation_rate,
            "avg_cpu_utilization": self.avg_cpu_utilization,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "load_imbalance": self.load_imbalance,
            "peak_valley_gap_kw": self.peak_valley_gap_kw,
            "total_tokens": self.total_tokens,
            "unit_token_energy_kwh_per_million": self.unit_token_energy_kwh_per_million,
            "unit_token_cost_per_million": self.unit_token_cost_per_million,
            "power_limit_violation_hours": self.power_limit_violation_hours,
            "peak_limit_violation_hours": self.peak_limit_violation_hours,
            "remote_task_count": self.remote_task_count,
            "remote_completion_rate": self.remote_completion_rate,
            "remote_cost": self.remote_cost,
            "remote_energy_kwh": self.remote_energy_kwh,
            "migration_delay_hours": self.migration_delay_hours,
            "total_completed_tasks": self.total_completed_tasks,
            "total_tasks": self.total_tasks,
            "objective_penalty": self.objective_penalty,
        }


def _clone_tasks_by_slot(tasks: list[Task], hours: int) -> list[list[TaskState]]:
    """将输入任务复制到各小时桶中，避免仿真过程修改原始任务对象。"""
    task_buckets: list[list[TaskState]] = [[] for _ in range(hours)]
    for task in tasks:
        task_buckets[int(task.arrival_time)].append(
            TaskState(
                task_id=task.task_id,
                arrival_slot=int(task.arrival_time),
                task_type=task.task_type,
                priority=int(task.priority),
                deadline_slot=min(int(task.deadline_slot), hours - 1),
                preferred_resource=task.preferred_resource,
                service_demand=float(task.service_demand),
                memory_demand=float(task.memory_demand),
                bandwidth_demand=float(task.bandwidth_demand),
                token_amount=float(task.token_amount),
                delay_tolerant=task.delay_tolerant,
                earliest_service_slot=int(task.arrival_time),
                migratable=bool(task.migratable),
                migration_cost_weight=float(task.migration_cost_weight),
                migration_delay_penalty=float(task.migration_delay_penalty),
            )
        )
    return task_buckets


def _defer_new_tasks(new_tasks: list[TaskState], hour: int, defer_ratio: float, high_price: bool, deadline_guard_slots: int) -> int:
    """在高电价时段延后部分可延迟任务，但保留接近截止时间的任务。"""
    if not high_price or defer_ratio <= 0.0:
        return 0

    candidates = [task for task in new_tasks if task.delay_tolerant and (task.deadline_slot - hour) > deadline_guard_slots]
    defer_count = int(len(candidates) * defer_ratio)
    for task in candidates[:defer_count]:
        task.earliest_service_slot = min(task.deadline_slot, hour + 1)
    return defer_count


def _has_local_capacity_pressure(
    slot_tasks: list[TaskState],
    cpu_queue: list[TaskState],
    gpu_queue: list[TaskState],
    hour: int,
    resource_pool: ResourcePool,
    active_cpu_servers: int,
    active_gpu_servers: int,
) -> bool:
    """判断本地 CPU/GPU 队列是否存在容量压力，作为空间迁移触发条件之一。"""
    candidate_tasks = [task for task in slot_tasks if hour >= task.earliest_service_slot]
    cpu_demand = sum(task.service_demand for task in cpu_queue if hour >= task.earliest_service_slot)
    cpu_demand += sum(task.service_demand for task in candidate_tasks if task.preferred_resource == "cpu")
    gpu_demand = sum(task.service_demand for task in gpu_queue if hour >= task.earliest_service_slot)
    gpu_demand += sum(task.service_demand for task in candidate_tasks if task.preferred_resource == "gpu")

    cpu_capacity = resource_pool.cpu.compute_capacity(active_cpu_servers)
    gpu_capacity = resource_pool.gpu.compute_capacity(active_gpu_servers)
    return cpu_demand > cpu_capacity * 0.9 or gpu_demand > gpu_capacity * 0.9


def _migrate_new_tasks(
    new_tasks: list[TaskState],
    hour: int,
    migration_ratio: float,
    high_price: bool,
    capacity_pressure: bool,
    migration_cfg: dict,
    slot_hours: float,
) -> tuple[list[TaskState], dict[str, float]]:
    """轻量空间迁移：把部分可迁移任务送到外部算力池，当期完成并计入迁移代价。"""
    if not migration_cfg.get("enable_spatial_migration", False) or migration_ratio <= 0.0:
        return new_tasks, _empty_remote_stats()

    high_price_only = bool(migration_cfg.get("high_price_only", True))
    should_migrate = high_price or capacity_pressure
    if high_price_only and not high_price:
        should_migrate = False
    if not should_migrate:
        return new_tasks, _empty_remote_stats()

    max_ratio = float(migration_cfg.get("max_migration_ratio", 0.0))
    ratio = max(0.0, min(float(migration_ratio), max_ratio))
    candidates = [
        task
        for task in new_tasks
        if task.migratable and hour >= task.earliest_service_slot
    ]
    migrate_count = int(len(candidates) * ratio)
    if migrate_count <= 0:
        return new_tasks, _empty_remote_stats()

    # 远端容量只做小时级抽象，不维护完整远端排队系统。
    remaining_cpu = float(migration_cfg.get("remote_capacity_cpu", 0.0))
    remaining_gpu = float(migration_cfg.get("remote_capacity_gpu", 0.0))
    base_delay = float(migration_cfg.get("migration_delay_hours", 0.0))
    cost_per_task = float(migration_cfg.get("migration_cost_per_task", 0.0))
    energy_per_task = float(migration_cfg.get("network_energy_kwh_per_task", 0.0))

    selected_ids: set[str] = set()
    completed = 0
    violations = 0
    remote_sensitive_violations = 0
    delay_sum = 0.0
    remote_cost = 0.0
    remote_energy = 0.0
    remote_tokens = 0.0
    remote_cpu_load = 0.0
    remote_gpu_load = 0.0

    # 优先迁移低优先级、token 类和更适合远端批处理的任务，避免时延敏感任务被过度迁移。
    sorted_candidates = sorted(candidates, key=lambda item: (item.priority, item.task_type != "token_batch", item.deadline_slot))
    for task in sorted_candidates[:migrate_count]:
        if task.preferred_resource == "gpu":
            if remaining_gpu < task.service_demand:
                continue
            remaining_gpu -= task.service_demand
            remote_gpu_load += task.service_demand
        else:
            if remaining_cpu < task.service_demand:
                continue
            remaining_cpu -= task.service_demand
            remote_cpu_load += task.service_demand

        task.executed_remotely = True
        selected_ids.add(task.task_id)
        completed += 1
        remote_tokens += task.token_amount
        remote_delay = base_delay + task.migration_delay_penalty
        delay_sum += remote_delay
        remote_cost += cost_per_task * task.migration_cost_weight
        remote_energy += energy_per_task
        deadline_allowance = max(0.0, (task.deadline_slot - hour + 1) * slot_hours)
        if remote_delay > deadline_allowance:
            violations += 1
            # 远端执行的时延敏感任务同样纳入敏感任务 SLA 违约统计。
            if task.task_type == "delay_sensitive":
                remote_sensitive_violations += 1

    remaining_tasks = [task for task in new_tasks if task.task_id not in selected_ids]
    return remaining_tasks, {
        "completed": float(completed),
        "violations": float(violations),
        "remote_sensitive_violations": float(remote_sensitive_violations),
        "delay_sum": delay_sum,
        "remote_cost": remote_cost,
        "remote_energy_kwh": remote_energy,
        "served_tokens": remote_tokens,
        "remote_cpu_load": remote_cpu_load,
        "remote_gpu_load": remote_gpu_load,
        "attempted": float(migrate_count),
    }


def _empty_remote_stats() -> dict[str, float]:
    """返回空迁移统计，减少主循环中的分支判断。"""
    return {
        "completed": 0.0,
        "violations": 0.0,
        "remote_sensitive_violations": 0.0,
        "delay_sum": 0.0,
        "remote_cost": 0.0,
        "remote_energy_kwh": 0.0,
        "served_tokens": 0.0,
        "remote_cpu_load": 0.0,
        "remote_gpu_load": 0.0,
        "attempted": 0.0,
    }


def _task_sort_key(task: TaskState, mode: str) -> tuple:
    """不同算法使用不同队列排序规则：FCFS、价格优先或综合优先级。"""
    if mode in {"fcfs", "no_priority"}:
        return (task.arrival_slot, task.task_id)
    if mode == "price_only":
        return (task.deadline_slot, -task.priority, task.arrival_slot, task.task_id)
    return (-task.priority, task.deadline_slot, task.arrival_slot, task.task_id)


def _dispatch_resource_queue(
    queue: list[TaskState],
    mode: str,
    current_hour: int,
    active_servers: int,
    resource_name: str,
    resource_pool: ResourcePool,
    slot_hours: float,
) -> tuple[list[TaskState], dict[str, float]]:
    """在单类资源队列上执行任务，返回未完成任务和该时隙统计量。"""
    resource = resource_pool.as_dict()[resource_name]
    compute_capacity = resource.compute_capacity(active_servers)
    memory_capacity = resource.memory_capacity_total(active_servers)
    bandwidth_capacity = resource.bandwidth_capacity_total(active_servers)

    eligible_tasks = [task for task in queue if current_hour >= task.earliest_service_slot]
    raw_queue_demand = sum(task.service_demand for task in eligible_tasks)
    stable_queue_demand = 0.0
    if active_servers > 0:
        # 跨时隙积压已由 waited_slots 计入；这里将时隙内排队估计限制在稳定区间，
        # 避免同时把积压等待和 M/M/c 不稳定惩罚重复计入平均时延。
        stable_queue_demand = min(raw_queue_demand, compute_capacity * 0.95)

    queue_delay = 0.0
    if stable_queue_demand > 0.0 and active_servers > 0:
        queue_delay = estimate_delay_statistics(
            arrival_rate=stable_queue_demand,
            service_rate=resource.queue_service_rate,
            servers=max(1, active_servers),
            waited_slots=0.0,
            slot_hours=slot_hours,
        )["within_slot_delay_hours"]
        queue_delay = min(queue_delay, slot_hours)

    completed = 0
    violations = 0
    sensitive_violations = 0
    delay_sum = 0.0
    served_load = 0.0
    served_tokens = 0.0
    remaining: list[TaskState] = []

    for task in sorted(queue, key=lambda item: _task_sort_key(item, mode)):
        if current_hour < task.earliest_service_slot:
            remaining.append(task)
            continue

        fits = (
            compute_capacity >= task.service_demand
            and memory_capacity >= task.memory_demand
            and bandwidth_capacity >= task.bandwidth_demand
        )
        if fits:
            compute_capacity -= task.service_demand
            memory_capacity -= task.memory_demand
            bandwidth_capacity -= task.bandwidth_demand
            completed += 1
            served_load += task.service_demand
            served_tokens += task.token_amount
            waited_slots = max(0, current_hour - task.arrival_slot)
            delay_sum += waited_slots * slot_hours + queue_delay
        else:
            remaining.append(task)

    post_dispatch: list[TaskState] = []
    for task in remaining:
        if task.deadline_slot <= current_hour:
            violations += 1
            if task.task_type == "delay_sensitive":
                sensitive_violations += 1
        else:
            post_dispatch.append(task)

    total_seen = completed + len(post_dispatch) + violations
    utilization = safe_divide(served_load, resource.compute_capacity(active_servers))
    completion_rate = safe_divide(completed, max(total_seen, 1))

    return post_dispatch, {
        "completed": completed,
        "violations": violations,
        "sensitive_violations": sensitive_violations,
        "delay_sum": delay_sum,
        "served_load": served_load,
        "served_tokens": served_tokens,
        "utilization": utilization,
        "completion_rate": completion_rate,
    }


def simulate_schedule(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    schedule: SchedulePlan,
    base_cfg: dict,
    price_cfg: dict,
    dispatch_mode: str,
) -> SimulationResult:
    """执行 24 小时调度仿真，汇总成本、时延、SLA、利用率和 token 指标。"""
    hours = int(base_cfg["time"]["hours"])
    slot_hours = float(base_cfg["time"]["slot_hours"])
    constraints = base_cfg["constraints"]
    power_cfg = base_cfg["power"]
    migration_cfg = base_cfg.get("migration", {})

    high_price_threshold = float(hourly_df["price"].quantile(price_cfg["high_price_quantile"]))
    task_buckets = _clone_tasks_by_slot(tasks, hours)
    migration_ratio = schedule.migration_ratio
    if migration_ratio.size == 0:
        migration_ratio = np.zeros(hours, dtype=float)

    cpu_queue: list[TaskState] = []
    gpu_queue: list[TaskState] = []

    total_completed = 0
    total_violations = 0
    sensitive_total = sum(task.task_type == "delay_sensitive" for task in tasks)
    sensitive_violations = 0
    delay_sum = 0.0
    total_tokens = 0.0
    deferred_count = 0
    remote_completed = 0
    remote_violations = 0
    remote_cost = 0.0
    remote_energy_kwh = 0.0
    remote_delay_sum = 0.0
    remote_attempted = 0.0

    hourly_power_kw: list[float] = []
    hourly_effective_power_kw: list[float] = []
    hourly_remote_energy_kwh: list[float] = []
    hourly_remote_task_count: list[int] = []
    hourly_delay_hours: list[float] = []
    hourly_completion_rate: list[float] = []
    hourly_cpu_utilization: list[float] = []
    hourly_gpu_utilization: list[float] = []
    power_limit_violation_hours = 0
    peak_limit_violation_hours = 0

    for hour in range(hours):
        slot_tasks = task_buckets[hour]
        is_high_price = float(hourly_df.loc[hour, "price"]) >= high_price_threshold
        # 高电价时段只延后可延迟任务；时延敏感任务仍直接进入队列。
        deferred_count += _defer_new_tasks(
            new_tasks=slot_tasks,
            hour=hour,
            defer_ratio=float(schedule.defer_ratio[hour]),
            high_price=is_high_price and dispatch_mode in {"price_only", "proposed", "no_priority"},
            deadline_guard_slots=1,
        )

        capacity_pressure = _has_local_capacity_pressure(
            slot_tasks=slot_tasks,
            cpu_queue=cpu_queue,
            gpu_queue=gpu_queue,
            hour=hour,
            resource_pool=resource_pool,
            active_cpu_servers=int(schedule.cpu_servers[hour]),
            active_gpu_servers=int(schedule.gpu_servers[hour]),
        )
        if dispatch_mode in {"proposed", "no_priority"}:
            # 空间迁移发生在新任务入本地队列之前；远端执行任务不再计入本地 IT 功率。
            slot_tasks, remote_stats = _migrate_new_tasks(
                new_tasks=slot_tasks,
                hour=hour,
                migration_ratio=float(migration_ratio[hour]),
                high_price=is_high_price,
                capacity_pressure=capacity_pressure,
                migration_cfg=migration_cfg,
                slot_hours=slot_hours,
            )
        else:
            remote_stats = _empty_remote_stats()

        remote_completed += int(remote_stats["completed"])
        remote_violations += int(remote_stats["violations"])
        remote_attempted += float(remote_stats["attempted"])
        remote_cost += float(remote_stats["remote_cost"])
        remote_energy_kwh += float(remote_stats["remote_energy_kwh"])
        remote_delay_sum += float(remote_stats["delay_sum"])
        delay_sum += float(remote_stats["delay_sum"])
        total_tokens += float(remote_stats["served_tokens"])

        for task in slot_tasks:
            # 当前最小可行版本采用不可分任务：一个任务只进入 CPU 或 GPU 中的一个队列。
            if task.preferred_resource == "gpu":
                gpu_queue.append(task)
            else:
                cpu_queue.append(task)

        cpu_queue, cpu_stats = _dispatch_resource_queue(
            queue=cpu_queue,
            mode=dispatch_mode,
            current_hour=hour,
            active_servers=int(schedule.cpu_servers[hour]),
            resource_name="cpu",
            resource_pool=resource_pool,
            slot_hours=slot_hours,
        )
        gpu_queue, gpu_stats = _dispatch_resource_queue(
            queue=gpu_queue,
            mode=dispatch_mode,
            current_hour=hour,
            active_servers=int(schedule.gpu_servers[hour]),
            resource_name="gpu",
            resource_pool=resource_pool,
            slot_hours=slot_hours,
        )

        slot_completed = int(cpu_stats["completed"] + gpu_stats["completed"])
        slot_violations = int(cpu_stats["violations"] + gpu_stats["violations"])
        total_completed += slot_completed + int(remote_stats["completed"])
        total_violations += slot_violations + int(remote_stats["violations"])
        sensitive_violations += int(
            cpu_stats["sensitive_violations"]
            + gpu_stats["sensitive_violations"]
            + remote_stats["remote_sensitive_violations"]
        )
        delay_sum += float(cpu_stats["delay_sum"] + gpu_stats["delay_sum"])
        total_tokens += float(cpu_stats["served_tokens"] + gpu_stats["served_tokens"])

        served_load = {"cpu": cpu_stats["served_load"], "gpu": gpu_stats["served_load"]}
        power_detail = hourly_heterogeneous_power_kw(
            resource_pool=resource_pool,
            active_servers={"cpu": int(schedule.cpu_servers[hour]), "gpu": int(schedule.gpu_servers[hour])},
            served_load=served_load,
            fixed_power_kw=float(power_cfg["fixed_power_kw"]),
            cooling_base_coeff=float(power_cfg["cooling_base_coeff"]),
        )
        total_power = power_detail["total_power_kw"]
        hourly_power_kw.append(total_power)
        hourly_remote_energy_kwh.append(float(remote_stats["remote_energy_kwh"]))
        hourly_remote_task_count.append(int(remote_stats["completed"]))
        hourly_effective_power_kw.append(total_power + float(remote_stats["remote_energy_kwh"]) / max(slot_hours, 1e-9))

        if total_power > float(constraints["power_limit_kw"]):
            power_limit_violation_hours += 1
        if hour in constraints["peak_hours"]:
            peak_limit = float(constraints["power_limit_kw"]) * float(constraints["peak_reduction_ratio"])
            if total_power > peak_limit:
                peak_limit_violation_hours += 1

        hourly_cpu_utilization.append(float(cpu_stats["utilization"]))
        hourly_gpu_utilization.append(float(gpu_stats["utilization"]))
        hourly_delay_hours.append(
            safe_divide(
                cpu_stats["delay_sum"] + gpu_stats["delay_sum"] + remote_stats["delay_sum"],
                max(slot_completed + int(remote_stats["completed"]), 1),
            )
        )
        hourly_completion_rate.append(
            safe_divide(
                slot_completed + int(remote_stats["completed"]),
                max(slot_completed + slot_violations + int(remote_stats["completed"]) + int(remote_stats["violations"]), 1),
            )
        )

    local_energy_kwh = float(sum(hourly_power_kw) * slot_hours)
    total_energy_kwh = local_energy_kwh + remote_energy_kwh
    local_electricity_cost = total_energy_cost(
        hourly_power_kw=hourly_power_kw,
        prices=hourly_df["price"].to_numpy(dtype=float),
        delta_t_hours=slot_hours,
    )
    total_carbon = total_carbon_emission(
        hourly_power_kw=hourly_power_kw,
        carbon_factors=hourly_df["carbon_factor"].to_numpy(dtype=float),
        delta_t_hours=slot_hours,
    )
    total_carbon += remote_energy_kwh * float(hourly_df["carbon_factor"].mean())

    renewable_credit = float(hourly_df["renewable_ratio"].mean()) * float(power_cfg["renewable_credit_factor"])
    # 绿电比例在这里作为电费折扣近似，碳成本参数默认为 0，可在配置中开启。
    total_cost = max(0.0, local_electricity_cost * (1.0 - renewable_credit)) + remote_cost
    total_cost += total_carbon * float(power_cfg["carbon_cost_per_kg"])

    avg_delay_hours = safe_divide(delay_sum, max(total_completed, 1))
    completion_rate = safe_divide(total_completed, len(tasks))
    sla_violation_rate = safe_divide(total_violations, len(tasks))
    delay_sensitive_violation_rate = safe_divide(sensitive_violations, max(sensitive_total, 1))

    load_imbalance = compute_load_imbalance(hourly_cpu_utilization, hourly_gpu_utilization)
    peak_valley_gap_kw = compute_peak_valley_gap(hourly_power_kw)

    penalty = 0.0
    # 约束惩罚项不会单独输出为目标，但会影响 NSGA-II 对不可行方案的排序。
    penalty += max(0.0, avg_delay_hours - float(constraints["max_avg_delay_hours"])) * 1000.0
    penalty += max(0.0, sla_violation_rate - float(constraints["max_sla_violation_rate"])) * 5000.0
    penalty += power_limit_violation_hours * 500.0
    penalty += peak_limit_violation_hours * 250.0
    penalty += delay_sensitive_violation_rate * 4000.0
    penalty += remote_completed * 0.02

    token_scale = float(base_cfg["task_defaults"]["token_energy_scale"])
    unit_token_energy = safe_divide(total_energy_kwh, max(total_tokens, 1.0) / token_scale)
    unit_token_cost = safe_divide(total_cost, max(total_tokens, 1.0) / token_scale)

    return SimulationResult(
        total_energy_kwh=total_energy_kwh,
        total_cost=total_cost,
        total_carbon=total_carbon,
        avg_delay_hours=avg_delay_hours,
        completion_rate=completion_rate,
        sla_violation_rate=sla_violation_rate,
        delay_sensitive_violation_rate=delay_sensitive_violation_rate,
        avg_cpu_utilization=float(np.mean(hourly_cpu_utilization)) if hourly_cpu_utilization else 0.0,
        avg_gpu_utilization=float(np.mean(hourly_gpu_utilization)) if hourly_gpu_utilization else 0.0,
        load_imbalance=load_imbalance,
        peak_valley_gap_kw=peak_valley_gap_kw,
        total_tokens=total_tokens,
        unit_token_energy_kwh_per_million=unit_token_energy,
        unit_token_cost_per_million=unit_token_cost,
        power_limit_violation_hours=power_limit_violation_hours,
        peak_limit_violation_hours=peak_limit_violation_hours,
        remote_task_count=remote_completed,
        remote_completion_rate=safe_divide(remote_completed, max(remote_attempted, 1.0)),
        remote_cost=remote_cost,
        remote_energy_kwh=remote_energy_kwh,
        migration_delay_hours=safe_divide(remote_delay_sum, max(remote_completed, 1)),
        total_completed_tasks=total_completed,
        total_tasks=len(tasks),
        objective_penalty=penalty,
        hourly_power_kw=hourly_power_kw,
        hourly_delay_hours=hourly_delay_hours,
        hourly_completion_rate=hourly_completion_rate,
        hourly_cpu_utilization=hourly_cpu_utilization,
        hourly_gpu_utilization=hourly_gpu_utilization,
        hourly_cpu_servers=schedule.cpu_servers.astype(int).tolist(),
        hourly_gpu_servers=schedule.gpu_servers.astype(int).tolist(),
        hourly_remote_energy_kwh=hourly_remote_energy_kwh,
        hourly_remote_task_count=hourly_remote_task_count,
        hourly_effective_power_kw=hourly_effective_power_kw,
        deferred_task_count=deferred_count,
    )
