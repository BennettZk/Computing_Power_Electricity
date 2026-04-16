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
    cpu_servers: np.ndarray
    gpu_servers: np.ndarray
    defer_ratio: np.ndarray


@dataclass
class TaskState:
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


@dataclass
class SimulationResult:
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
            "total_completed_tasks": self.total_completed_tasks,
            "total_tasks": self.total_tasks,
            "objective_penalty": self.objective_penalty,
        }


def _clone_tasks_by_slot(tasks: list[Task], hours: int) -> list[list[TaskState]]:
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
            )
        )
    return task_buckets


def _defer_new_tasks(new_tasks: list[TaskState], hour: int, defer_ratio: float, high_price: bool, deadline_guard_slots: int) -> int:
    if not high_price or defer_ratio <= 0.0:
        return 0

    candidates = [task for task in new_tasks if task.delay_tolerant and (task.deadline_slot - hour) > deadline_guard_slots]
    defer_count = int(len(candidates) * defer_ratio)
    for task in candidates[:defer_count]:
        task.earliest_service_slot = min(task.deadline_slot, hour + 1)
    return defer_count


def _task_sort_key(task: TaskState, mode: str) -> tuple:
    if mode == "fcfs":
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
    resource = resource_pool.as_dict()[resource_name]
    compute_capacity = resource.compute_capacity(active_servers)
    memory_capacity = resource.memory_capacity_total(active_servers)
    bandwidth_capacity = resource.bandwidth_capacity_total(active_servers)

    eligible_tasks = [task for task in queue if current_hour >= task.earliest_service_slot]
    raw_queue_demand = sum(task.service_demand for task in eligible_tasks)
    stable_queue_demand = 0.0
    if active_servers > 0:
        # Cross-slot backlog is already counted via waited_slots. Keep the within-slot
        # queueing estimate in a stable regime to avoid double counting unstable bursts.
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
    hours = int(base_cfg["time"]["hours"])
    slot_hours = float(base_cfg["time"]["slot_hours"])
    constraints = base_cfg["constraints"]
    power_cfg = base_cfg["power"]

    high_price_threshold = float(hourly_df["price"].quantile(price_cfg["high_price_quantile"]))
    task_buckets = _clone_tasks_by_slot(tasks, hours)

    cpu_queue: list[TaskState] = []
    gpu_queue: list[TaskState] = []

    total_completed = 0
    total_violations = 0
    sensitive_total = sum(task.task_type == "delay_sensitive" for task in tasks)
    sensitive_violations = 0
    delay_sum = 0.0
    total_tokens = 0.0
    deferred_count = 0

    hourly_power_kw: list[float] = []
    hourly_delay_hours: list[float] = []
    hourly_completion_rate: list[float] = []
    hourly_cpu_utilization: list[float] = []
    hourly_gpu_utilization: list[float] = []
    power_limit_violation_hours = 0
    peak_limit_violation_hours = 0

    for hour in range(hours):
        slot_tasks = task_buckets[hour]
        is_high_price = float(hourly_df.loc[hour, "price"]) >= high_price_threshold
        deferred_count += _defer_new_tasks(
            new_tasks=slot_tasks,
            hour=hour,
            defer_ratio=float(schedule.defer_ratio[hour]),
            high_price=is_high_price and dispatch_mode in {"price_only", "proposed"},
            deadline_guard_slots=1,
        )

        for task in slot_tasks:
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
        total_completed += slot_completed
        total_violations += slot_violations
        sensitive_violations += int(cpu_stats["sensitive_violations"] + gpu_stats["sensitive_violations"])
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

        if total_power > float(constraints["power_limit_kw"]):
            power_limit_violation_hours += 1
        if hour in constraints["peak_hours"]:
            peak_limit = float(constraints["power_limit_kw"]) * float(constraints["peak_reduction_ratio"])
            if total_power > peak_limit:
                peak_limit_violation_hours += 1

        hourly_cpu_utilization.append(float(cpu_stats["utilization"]))
        hourly_gpu_utilization.append(float(gpu_stats["utilization"]))
        hourly_delay_hours.append(safe_divide(cpu_stats["delay_sum"] + gpu_stats["delay_sum"], max(slot_completed, 1)))
        hourly_completion_rate.append(safe_divide(slot_completed, max(slot_completed + slot_violations, 1)))

    total_energy_kwh = float(sum(hourly_power_kw) * slot_hours)
    total_cost = total_energy_cost(
        hourly_power_kw=hourly_power_kw,
        prices=hourly_df["price"].to_numpy(dtype=float),
        delta_t_hours=slot_hours,
    )
    total_carbon = total_carbon_emission(
        hourly_power_kw=hourly_power_kw,
        carbon_factors=hourly_df["carbon_factor"].to_numpy(dtype=float),
        delta_t_hours=slot_hours,
    )

    renewable_credit = float(hourly_df["renewable_ratio"].mean()) * float(power_cfg["renewable_credit_factor"])
    total_cost = max(0.0, total_cost * (1.0 - renewable_credit))
    total_cost += total_carbon * float(power_cfg["carbon_cost_per_kg"])

    avg_delay_hours = safe_divide(delay_sum, max(total_completed, 1))
    completion_rate = safe_divide(total_completed, len(tasks))
    sla_violation_rate = safe_divide(total_violations, len(tasks))
    delay_sensitive_violation_rate = safe_divide(sensitive_violations, max(sensitive_total, 1))

    load_imbalance = compute_load_imbalance(hourly_cpu_utilization, hourly_gpu_utilization)
    peak_valley_gap_kw = compute_peak_valley_gap(hourly_power_kw)

    penalty = 0.0
    penalty += max(0.0, avg_delay_hours - float(constraints["max_avg_delay_hours"])) * 1000.0
    penalty += max(0.0, sla_violation_rate - float(constraints["max_sla_violation_rate"])) * 5000.0
    penalty += power_limit_violation_hours * 500.0
    penalty += peak_limit_violation_hours * 250.0
    penalty += delay_sensitive_violation_rate * 4000.0

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
        deferred_task_count=deferred_count,
    )
