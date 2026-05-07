from __future__ import annotations

import numpy as np
import pandas as pd

from models.resource import ResourcePool
from models.task import Task
from optimizers.common import SingleObjectiveRunResult, build_bounds, evaluate_weighted_fitness


def _evaluate_population(
    population: np.ndarray,
    bounds,
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
) -> tuple[np.ndarray, list[tuple]]:
    fitness_values: list[float] = []
    evaluated: list[tuple] = []
    for individual in population:
        fitness, schedule, metrics = evaluate_weighted_fitness(
            vector=individual,
            bounds=bounds,
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
        )
        fitness_values.append(fitness)
        evaluated.append((fitness, schedule, metrics))
    return np.asarray(fitness_values, dtype=float), evaluated


def _tournament_select(population: np.ndarray, fitness_values: np.ndarray, rng: np.random.Generator, tournament_size: int) -> np.ndarray:
    indices = rng.integers(0, len(population), size=tournament_size)
    best_idx = indices[int(np.argmin(fitness_values[indices]))]
    return population[best_idx].copy()


def _mutate_integer_vector(
    individual: np.ndarray,
    bounds,
    rng: np.random.Generator,
    mutation_rate: float,
) -> np.ndarray:
    mask = rng.random(individual.size) < mutation_rate
    if not np.any(mask):
        return individual
    mutation_indices = np.where(mask)[0]
    for idx in mutation_indices:
        individual[idx] = rng.integers(bounds.xl[idx], bounds.xu[idx] + 1)
    return individual


def run_ga(
    hourly_df: pd.DataFrame,
    tasks: list[Task],
    resource_pool: ResourcePool,
    base_cfg: dict,
    price_cfg: dict,
    experiment_cfg: dict,
) -> SingleObjectiveRunResult:
    """单目标遗传算法：使用与 NSGA-II 相同编码，最小化加权综合适应度。"""
    optimizer_cfg = experiment_cfg["optimizer"]
    bounds = build_bounds(resource_pool, base_cfg, experiment_cfg)
    n_var = bounds.xl.size
    pop_size = max(4, int(optimizer_cfg.get("ga_pop_size", 30)))
    n_gen = max(1, int(optimizer_cfg.get("ga_n_gen", 20)))
    crossover_prob = float(optimizer_cfg.get("ga_crossover_prob", 0.85))
    mutation_rate = float(optimizer_cfg.get("ga_mutation_rate", 1.0 / max(n_var, 1)))
    tournament_size = max(2, int(optimizer_cfg.get("ga_tournament_size", 3)))
    rng = np.random.default_rng(int(experiment_cfg["seed"]) + 101)

    population = rng.integers(bounds.xl, bounds.xu + 1, size=(pop_size, n_var))
    best_fitness = float("inf")
    best_schedule = None
    best_metrics = None
    best_vector = population[0].copy()
    convergence_rows: list[dict] = []

    for generation in range(n_gen):
        fitness_values, evaluated = _evaluate_population(
            population=population,
            bounds=bounds,
            hourly_df=hourly_df,
            tasks=tasks,
            resource_pool=resource_pool,
            base_cfg=base_cfg,
            price_cfg=price_cfg,
        )
        generation_best_idx = int(np.argmin(fitness_values))
        generation_best = evaluated[generation_best_idx]
        if generation_best[0] < best_fitness:
            best_fitness = float(generation_best[0])
            best_schedule = generation_best[1]
            best_metrics = generation_best[2]
            best_vector = population[generation_best_idx].copy()

        convergence_rows.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "generation_best_fitness": float(fitness_values[generation_best_idx]),
                "avg_fitness": float(np.mean(fitness_values)),
            }
        )

        next_population = [best_vector.copy()]
        while len(next_population) < pop_size:
            parent_a = _tournament_select(population, fitness_values, rng, tournament_size)
            parent_b = _tournament_select(population, fitness_values, rng, tournament_size)
            child_a = parent_a.copy()
            child_b = parent_b.copy()
            if rng.random() < crossover_prob:
                mask = rng.random(n_var) < 0.5
                child_a[mask], child_b[mask] = child_b[mask].copy(), child_a[mask].copy()
            next_population.append(_mutate_integer_vector(child_a, bounds, rng, mutation_rate))
            if len(next_population) < pop_size:
                next_population.append(_mutate_integer_vector(child_b, bounds, rng, mutation_rate))
        population = np.asarray(next_population, dtype=int)

    if best_schedule is None or best_metrics is None:
        raise RuntimeError("GA failed to evaluate any feasible individual.")

    metrics_dict = {**best_metrics.to_dict(), "fitness": best_fitness}
    return SingleObjectiveRunResult(
        best_schedule=best_schedule,
        best_metrics=metrics_dict,
        convergence_df=pd.DataFrame(convergence_rows),
    )
