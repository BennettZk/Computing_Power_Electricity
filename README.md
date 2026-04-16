# Heterogeneous Data Center Scheduling

This project incrementally refactors the original time-of-use demand response reproduction into a heterogeneous single-data-center scheduling study.

## What It Does

The code keeps the original paper's main line:

- time-of-use electricity pricing
- energy cost optimization
- delay and SLA control
- demand response
- multi-objective optimization

It extends the original homogeneous setting into:

- CPU and GPU heterogeneous resource pools
- delay-sensitive, delay-tolerant and token-batch task types
- power cap and peak shaving constraints
- baseline comparison against FCFS, Price-Only, Homogeneous-Baseline and Proposed

## How To Run

```powershell
.\.venv\Scripts\python.exe main.py
```

The entry script will:

1. generate processed hourly data if needed
2. generate synthetic task arrivals if needed
3. run all 4 algorithms
4. export tables, summary text and figures

## Input Data Format

Hourly profile:

- `data/processed/hourly_profile.csv`
- columns: `hour`, `arrival_rate`, `price`, `carbon_factor`, `renewable_ratio`

Task sequence:

- `data/synthetic/tasks.csv`
- columns: `task_id`, `arrival_time`, `task_type`, `cpu_demand`, `gpu_demand`, `memory_demand`, `bandwidth_demand`, `token_amount`, `deadline`, `priority`

## Outputs

Tables:

- `results.csv`
- `summary.txt`
- `outputs/proposed_pareto.csv`

Charts:

- `outputs/price_load_curve.png`
- `outputs/pareto_front.png`
- `outputs/energy_bar.png`
- `outputs/cpu_gpu_utilization.png`
- `outputs/convergence_curve.png`

## Algorithms

- `FCFS`: serves tasks in arrival order and ignores price response
- `Price-Only`: performs peak shaving against price without deep heterogeneous coordination
- `Homogeneous-Baseline`: keeps the original homogeneous-server assumption and optimizes a single server-count trajectory
- `Proposed`: coordinates CPU/GPU activation, delay-tolerant deferral and queue-delay control with NSGA-II
