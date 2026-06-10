from __future__ import annotations

from optimizers.nsga2 import run_nsga2


def run_proposed_scheduler(hourly_df, tasks, resource_pool, base_cfg, price_cfg, experiment_cfg):
    """运行本文提出的 NSGA-II 联合调度器。"""
    return run_nsga2(
        hourly_df=hourly_df,
        tasks=tasks,
        resource_pool=resource_pool,
        base_cfg=base_cfg,
        price_cfg=price_cfg,
        experiment_cfg=experiment_cfg,
    )
