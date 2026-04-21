# 异构资源环境下数据中心算电协同调度优化研究

本项目是在“面向电价型需求响应的数据中心能耗多目标联合优化策略”复现代码基础上做的增量式改造。当前版本保留原论文的分时电价、能耗成本、任务时延、需求响应和多目标优化主线，并将“同构服务器 + 单一负载”扩展为“CPU/GPU 异构资源池 + 多类型任务 + 时空迁移调度”。

这里的“空间迁移”不是完整多数据中心建模，也不是云边协同大系统，而是新增一个轻量的外部算力池 `remote_pool` 抽象。Proposed 调度器在高电价、本地功率压力较高或本地容量不足时，可以将部分可迁移任务转移到远端执行，从而形成“本地时间延迟 + 远端空间迁移”的折中。

## 项目解决的问题

给定 24 小时任务到达、电价、碳因子和数据中心 CPU/GPU 资源配置，本项目比较不同调度策略在成本、时延、能耗、SLA 和资源利用率上的表现。

核心问题可以概括为：在分时电价和异构资源约束下，如何同时决定每小时 CPU/GPU 开机数量、可延迟任务的时间迁移比例、可迁移任务的空间迁移比例，使数据中心在电费、时延、SLA、负载均衡和远端迁移代价之间取得折中。

## 目录结构

```text
dc_token_opt/
├─ config/
│  ├─ base.yaml              # 基础约束、SLA、功率上限、空间迁移参数
│  ├─ price.yaml             # 分时电价输入路径和高低电价分位数
│  ├─ resource.yaml          # CPU/GPU 异构资源池参数
│  └─ experiment.yaml        # 随机种子、输出路径、NSGA-II 参数
├─ data/
│  ├─ hourly_input.csv       # 小时级负载、电价、碳因子输入
│  ├─ processed/             # 自动生成的小时级曲线
│  └─ synthetic/             # 自动生成的任务序列
├─ models/
│  ├─ task.py                # 多类型任务模型与可迁移任务生成
│  ├─ resource.py            # CPU/GPU 资源模型
│  ├─ power_model.py         # 本地 IT 功率、制冷功率和成本计算
│  ├─ delay_model.py         # CPU/GPU 队列时延近似模型
│  └─ objective.py           # 调度仿真、时空迁移逻辑、指标汇总
├─ schedulers/
│  ├─ baseline_fcfs.py       # FCFS 基线
│  ├─ baseline_price_only.py # 仅电价响应基线
│  ├─ homogeneous_baseline.py# 同构服务器基线
│  └─ proposed_scheduler.py  # Proposed 调度器入口
├─ optimizers/
│  ├─ nsga2.py               # NSGA-II 多目标优化
│  ├─ ga.py                  # GA 预留接口
│  └─ pso.py                 # PSO 预留接口
├─ experiments/
│  ├─ exp_main.py            # 主实验入口
│  ├─ exp_ablation.py        # 消融实验
│  └─ exp_sensitivity.py     # 灵敏度实验
├─ utils/
│  ├─ io_utils.py            # 配置、数据和输出目录工具
│  ├─ metrics.py             # 指标表整理
│  ├─ plotting.py            # 图表绘制
│  └─ seed.py                # 随机种子设置
├─ main.py                   # 一键运行入口
├─ run_nsga2.py              # 兼容旧入口
├─ results.csv               # 主实验结果表
└─ summary.txt               # 实验摘要
```

## 运行环境

推荐直接使用项目已有虚拟环境运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

如果需要重新安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

主要依赖在 `requirements.txt` 中：

```text
numpy
pandas
matplotlib
scipy
pymoo
```

## 怎么运行

在 PowerShell 中进入项目根目录：

```powershell
cd E:\Coding\redo\dc_token_opt
```

运行主实验：

```powershell
.\.venv\Scripts\python.exe main.py
```

也可以运行兼容旧入口：

```powershell
.\.venv\Scripts\python.exe run_nsga2.py
```

主流程会依次执行四种算法对比、Proposed 消融实验、灵敏度实验、结果表格导出和图表绘制。

## 输入数据格式

默认小时级输入文件：

```text
data/hourly_input.csv
```

字段说明：

- `hour`：小时编号，范围为 0 到 23。
- `arrival_rate`：该小时任务到达强度。
- `price`：该小时分时电价。
- `carbon_factor`：该小时碳排放因子。

默认任务序列文件：

```text
data/synthetic/tasks.csv
```

如果文件不存在，程序会根据小时级负载自动生成合成任务。当前版本也会检查任务文件是否包含空间迁移字段；如果旧任务文件缺少这些字段，会自动重新生成，保证实验能体现空间迁移能力。

任务字段说明：

- `task_id`：任务编号。
- `arrival_time`：任务到达时隙。
- `task_type`：任务类型，包括 `delay_sensitive`、`delay_tolerant`、`token_batch`。
- `cpu_demand`：CPU 计算需求。
- `gpu_demand`：GPU 计算需求。
- `memory_demand`：内存需求。
- `bandwidth_demand`：带宽需求。
- `token_amount`：AI token 工作量，非 token 任务可为 0。
- `deadline`：任务截止时隙。
- `priority`：任务优先级，数值越大优先级越高。
- `migratable`：是否允许空间迁移到外部算力池。
- `migration_cost_weight`：迁移成本权重。
- `migration_delay_penalty`：迁移额外时延惩罚。

## 四种算法说明

`FCFS`：先来先服务基线。它根据任务到达顺序执行任务，不考虑分时电价，也不启用空间迁移。

`Price-Only`：仅电价响应基线。高电价时段压降部分可延迟负载，低电价时段释放负载，但不做异构资源联合优化，也不启用空间迁移。

`Homogeneous-Baseline`：同构服务器基线。它沿用原论文“同构服务器 + 单一负载”的建模思路，将 CPU/GPU 资源折算为等效服务器，用于对比异构感知调度的收益。

`Proposed`：本文方法。它使用 NSGA-II 同时优化每小时 CPU 开机数、GPU 开机数、时间迁移比例 `defer_ratio` 和空间迁移比例 `migration_ratio`。调度目标包括总成本、平均时延/SLA 和资源协同程度。空间迁移通过外部算力池抽象实现，远端执行任务不计入本地 IT 功率，但会产生远端执行成本、网络能耗和迁移时延惩罚。

## 时空迁移建模说明

时间迁移：对可延迟任务，在高电价时段按 `defer_ratio` 延后到后续时段处理，降低峰时本地用电压力。

空间迁移：对 `migratable=True` 的任务，在高电价、本地功率压力较高或本地 CPU/GPU 容量不足时，按 `migration_ratio` 尝试迁移到外部算力池执行。

外部算力池由 `config/base.yaml` 中的 `migration` 参数控制：

```json
"migration": {
  "enable_spatial_migration": true,
  "max_migration_ratio": 0.4,
  "remote_capacity_cpu": 12.0,
  "remote_capacity_gpu": 8.0,
  "migration_delay_hours": 0.2,
  "migration_cost_per_task": 0.08,
  "network_energy_kwh_per_task": 0.01,
  "high_price_only": true
}
```

如果将 `enable_spatial_migration` 改为 `false`，Proposed 会退化为原来的仅时间迁移版本。

## 输出结果

主实验运行后会生成：

```text
results.csv
summary.txt
outputs/proposed_pareto.csv
```

图表输出目录：

```text
outputs/
```

主要图表包括：

- `price_load_curve.png`：电价与本地负荷曲线。
- `pareto_front.png`：Proposed 时延-成本帕累托前沿。
- `energy_bar.png`：各算法总能耗柱状图。
- `cpu_gpu_utilization.png`：CPU/GPU 利用率对比图。
- `convergence_curve.png`：NSGA-II 收敛曲线。
- `spatial_migration_bar.png`：各算法远端迁移任务数对比。
- `load_shift_curve.png`：原始本地负荷与时空迁移后本地负荷对比。
- `time_space_ablation.png`：无迁移、仅时间迁移、时间+空间迁移的成本与时延对比。
- `ablation_results.png`：消融实验成本与时延对比。
- `sensitivity_results.png`：灵敏度实验成本与 SLA 对比。

扩展实验表格：

- `outputs/ablation_results.csv`：消融实验结果。
- `outputs/sensitivity_results.csv`：灵敏度实验结果。

## 结果字段说明

`results.csv` 和 `outputs/proposed_pareto.csv` 中包含常规指标：

- `total_energy_kwh`：总能耗，包含本地能耗和远端网络能耗。
- `total_cost`：总成本，包含本地电费、远端迁移成本和可选碳成本。
- `avg_delay_hours`：平均任务时延，包含远端迁移时延。
- `sla_violation_rate`：SLA 违约率。
- `cpu_utilization`：CPU 平均利用率。
- `gpu_utilization`：GPU 平均利用率。
- `load_imbalance`：负载不均衡度。
- `peak_valley_diff_kw`：本地功率峰谷差。
- `energy_per_million_tokens`：单位百万 token 能耗。
- `cost_per_million_tokens`：单位百万 token 成本。

新增空间迁移字段：

- `migration_ratio`：NSGA-II 染色体中的每小时空间迁移比例。
- `remote_task_count`：迁移到外部算力池执行的任务数。
- `remote_completion_rate`：可迁移任务中实际远端完成的比例。
- `remote_cost`：远端迁移执行成本。
- `remote_energy_kwh`：迁移网络能耗。
- `migration_delay_hours`：远端任务平均迁移时延。

## 消融实验

当前消融实验包含：

- `完整Proposed`：异构感知 + 电价响应 + 时间迁移 + 空间迁移 + 优先级。
- `无空间迁移`：关闭 `migration_ratio`，只保留时间迁移。
- `无时间迁移`：关闭 `defer_ratio`，保留空间迁移。
- `无异构感知`：使用同构服务器 baseline。
- `无优先级调度`：使用 FCFS 顺序评估同一开机方案。

消融结果会导出到：

```text
outputs/ablation_results.csv
outputs/time_space_ablation.png
```

## 灵敏度实验

当前灵敏度实验包含：

- `基准场景`
- `低负载0.8x`
- `高负载1.2x`
- `GPU数量减半`
- `功率上限收紧`
- `低远端容量`
- `高远端容量`
- `低迁移成本`
- `高迁移成本`
- `高电价波动`

灵敏度结果会导出到：

```text
outputs/sensitivity_results.csv
outputs/sensitivity_results.png
```

## 参数在哪里修改

常用参数集中在 `config/` 目录：

- 修改 CPU/GPU 服务器数量、功率、服务率：`config/resource.yaml`
- 修改 SLA、功率上限、峰时削减、空间迁移参数：`config/base.yaml`
- 修改 NSGA-II 种群规模、迭代代数、输出路径：`config/experiment.yaml`
- 修改电价输入路径和高低电价阈值：`config/price.yaml`

当前 `.yaml` 文件使用 JSON 兼容写法，因此可以用普通文本编辑器直接修改。

## 模型简化说明

当前版本做了以下简化，目的是保证本科论文实验可解释、可运行、可出图：

- 仍以单数据中心为主场景，不扩展为完整跨区域多数据中心系统。
- 空间迁移只通过外部算力池容量、迁移成本、网络能耗和迁移时延抽象实现。
- 远端不维护复杂队列，不建模真实网络拓扑，不模拟链上交易。
- 远端容量足够时，迁移任务视为当期或短延迟完成；远端容量不足时，任务回退到本地队列。
- token 只作为 AI 类任务工作量指标，用于计算单位 token 能耗和单位 token 成本。
- CPU/GPU 队列分别使用 M/M/c 近似估计时延。
- 制冷功率由 CPU/GPU IT 功率乘以资源类型制冷系数近似。

这些简化不会改变论文主线：项目仍围绕分时电价、需求响应、异构资源、任务时延和多目标优化展开。

## 常见问题

如果运行后没有图表，先确认命令是否在项目根目录执行：

```powershell
pwd
```

如果缺少依赖，重新安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果想重新生成任务数据，可以删除：

```text
data/synthetic/tasks.csv
```

然后重新运行：

```powershell
.\.venv\Scripts\python.exe main.py
```
