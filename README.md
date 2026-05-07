# 异构资源环境下数据中心算电协同调度优化研究

本项目是在“面向电价型需求响应的数据中心能耗多目标联合优化策略”复现代码基础上做的增量式改造。当前版本保留原论文的分时电价、能耗成本、任务时延、需求响应和多目标优化主线，并将“同构服务器 + 单一负载”扩展为“CPU/GPU 异构资源池 + 多类型任务 + 时空迁移调度”。

本项目的核心亮点不是完整多数据中心建模，而是在单数据中心异构资源调度场景下，引入轻量 `remote_pool` 抽象，刻画任务跨区迁移对成本、能耗和 SLA 的影响。Proposed 调度器在高电价、本地功率压力较高或本地容量不足时，可以将部分可迁移任务转移到远端执行，从而形成“本地时间延迟 + 远端空间迁移”的折中。

## 项目解决的问题

给定 24 小时任务到达、电价、碳因子和数据中心 CPU/GPU 资源配置，本项目比较不同调度策略在成本、时延、能耗、SLA 和资源利用率上的表现。

核心问题可以概括为：在分时电价和异构资源约束下，如何同时决定每小时 CPU/GPU 开机数量、可延迟任务的时间迁移比例、可迁移任务的空间迁移比例，使数据中心在电费、时延、SLA、负载均衡和远端迁移代价之间取得折中。

## 目录结构

```text
dc_token_opt/
├─ config/
│  ├─ base.yaml              # 基础约束、业务侧约束、功率上限、空间迁移参数
│  ├─ price.yaml             # 分时电价输入路径和高低电价分位数
│  ├─ resource.yaml          # CPU/GPU 异构资源池参数
│  └─ experiment.yaml        # 随机种子、输出路径、优化器参数
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
│  ├─ common.py              # GA/PSO 共享编码、解码和适应度函数
│  ├─ ga.py                  # 单目标遗传算法对比优化器
│  └─ pso.py                 # 单目标粒子群对比优化器
├─ experiments/
│  ├─ exp_main.py            # 主实验入口
│  ├─ exp_ablation.py        # 消融实验
│  ├─ exp_sensitivity.py     # 灵敏度实验
│  └─ exp_rolling.py         # Rolling-Proposed 滚动窗口实验
├─ scripts/
│  └─ prepare_real_dataset.py# 真实数据字段画像、清洗和标准格式转换
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

主流程会依次执行基线算法、GA/PSO 启发式算法、Proposed 主方法对比、Proposed 消融实验、Rolling-Proposed 滚动窗口实验、灵敏度实验、结果表格导出和图表绘制。

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

如果文件不存在，程序会根据小时级负载自动生成合成任务。任务加载函数兼容旧 CSV：如果旧任务文件缺少空间迁移字段，会为 `migratable`、`migration_cost_weight`、`migration_delay_penalty` 补默认值，不会因为缺字段自动重生成。若希望使用包含空间迁移属性分布的新合成任务数据，请手动删除 `data/synthetic/tasks.csv` 后重新运行。

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

## 真实数据接入与清洗

拿到真实数据后，建议先用 profile-only 模式查看字段结构，不直接改主实验输入：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_real_dataset.py --input data/raw/real_dataset.csv --profile-only
```

脚本会输出：

```text
outputs/real_data_profile.txt
outputs/real_data_cleaning_report.txt
```

确认时间列、任务 ID、CPU/GPU/内存和持续时间字段后，再用字段映射模式生成项目标准 CSV：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_real_dataset.py `
  --input data/raw/real_dataset.csv `
  --output-tasks data/real/tasks.csv `
  --output-hourly data/real/hourly_input.csv `
  --time-col submit_time `
  --task-id-col job_id `
  --cpu-col cpu_request `
  --gpu-col gpu_request `
  --memory-col memory_request `
  --duration-col duration
```

字段缺失时脚本会写入 warning 并使用默认值，例如 `gpu_demand=0`、`memory_demand=4.0`、`bandwidth_demand=0.5`、`deadline=arrival_time+2`。生成的 `tasks.csv` 会包含 `models.task.load_tasks_csv` 所需字段。

最后将 `config/experiment.yaml` 中 `paths.tasks_path` 改为：

```json
"tasks_path": "data/real/tasks.csv"
```

如需使用真实小时输入，可将 `data/real/hourly_input.csv` 复制到 `data/hourly_input.csv`，或在 `config/price.yaml` 中将 `legacy_profile_path` 指向 `data/real/hourly_input.csv`。如果已经生成过 `data/processed/hourly_profile.csv`，需要删除该文件后重新运行，确保新的小时曲线生效。

### 上传 XLSX 案例数据清洗

如果原始数据是以下三个 Excel 文件：

```text
data/raw/数据中心集群级别的调度数据.xlsx
data/raw/服务器级别的调度数据.xlsx
data/raw/芯片级别的调度数据.xlsx
```

可以运行专用清洗脚本：

```powershell
python scripts/prepare_uploaded_case_dataset.py
```

默认只生成 `data/real_case/` 下的标准 CSV，不会替换主实验默认输入：

```text
data/real_case/server_tasks_24h.csv
data/real_case/server_tasks_5min_raw_mapped.csv
data/real_case/cluster_power_price_5min.csv
data/real_case/hourly_input_24h.csv
data/real_case/chip_dvfs.csv
outputs/uploaded_case_cleaning_report.txt
```

其中 `server_tasks_24h.csv` 兼容 `models.task.load_tasks_csv`，`hourly_input_24h.csv` 兼容项目小时输入格式；芯片级 `chip_dvfs.csv` 只用于拓展分析，不进入 `main.py` 主输入。

如果确认清洗结果无误，可以选择替换主实验输入：

```powershell
python scripts/prepare_uploaded_case_dataset.py --replace-main-inputs
```

该模式会先备份 `data/synthetic/tasks.csv` 和 `data/hourly_input.csv`，再复制清洗后的真实案例输入，并删除 `data/processed/hourly_profile.csv` 缓存，避免继续读取旧小时曲线。默认模式不会执行这些替换。

## 真实集群级数据的保留与使用方式

上传的真实案例数据是集群级调度数据，原始任务规模远高于本项目默认的单数据中心等效资源规模。如果直接把全量真实任务接入默认 `main.py`，可能出现 SLA 违约率接近 1、任务完成率很低、空间迁移和 GPU/token 指标不明显的结果。这是资源规模和任务规模不匹配造成的，不代表程序崩溃，也不应作为论文主实验算法优劣结论。

项目保留三种真实数据场景：

- `real_5k`：真实负载趋势缩放版，用于快速验证。
- `real_10k`：真实负载趋势增强版，用于扩展实验。
- `real_full`：原始全量真实数据，用于压力测试和模型适用边界分析。

生成真实缩放场景：

```powershell
python scripts/build_real_scenarios.py --target-sizes 5000 10000 full
```

也可以显式指定输入：

```powershell
python scripts/build_real_scenarios.py --input data/real_case/server_tasks_24h.csv --target-sizes 5000 10000 full
```

脚本会生成：

```text
data/real_scaled/tasks_5k.csv
data/real_scaled/tasks_10k.csv
data/real_scaled/tasks_full.csv
data/real_scaled/hourly_input_5k.csv
data/real_scaled/hourly_input_10k.csv
data/real_scaled/hourly_input_full.csv
outputs/real_scaled_report.txt
```

这些缩放版任务会保留真实小时级负载趋势，同时补齐 GPU、Token、可迁移任务和迁移时延字段，使 CPU/GPU 异构、Token 任务、空间迁移和跨区时延分析能够体现出来。默认 `main.py` 仍使用 `data/synthetic/tasks.csv` 和 `data/hourly_input.csv`，不会自动切换到真实缩放数据。

推荐运行流程：

```powershell
python scripts/build_real_scenarios.py
python run_real_quick.py --scenario 5k
python run_real_quick.py --scenario 10k
python run_real_quick.py --scenario full --stress
```

也可以运行压力测试入口：

```powershell
python run_real_stress.py
```

真实 quick 输出位于：

```text
outputs/real_quick/results_real_quick.csv
outputs/real_quick/results_real_quick_cn.csv
outputs/real_quick/summary_real_quick.txt
outputs/real_quick/energy_bar.png
outputs/real_quick/cpu_gpu_utilization.png
outputs/real_quick/spatial_migration_bar.png
```

真实 stress 输出位于：

```text
outputs/real_stress/results_real_stress.csv
outputs/real_stress/results_real_stress_cn.csv
outputs/real_stress/summary_real_stress.txt
```

`results_real_quick.csv` 和 `results_real_stress.csv` 使用英文表头，便于后续程序继续读取；带 `_cn.csv` 后缀的是中文展示版，不要把中文表头结果文件作为程序输入。

论文中建议表述：

```text
本文基于真实集群级数据提取小时级负载波动趋势，并构建与单数据中心资源规模相匹配的缩放实验场景；原始全量数据用于压力测试，以分析模型在极端高负载条件下的适用边界。
```

## 算法说明

`FCFS`：先来先服务基线。它根据任务到达顺序执行任务，不考虑分时电价，也不启用空间迁移。

`Price-Only`：仅电价响应基线。高电价时段压降部分可延迟负载，低电价时段释放负载，但不做异构资源联合优化，也不启用空间迁移。

`Homogeneous-Baseline`：同构服务器基线。它沿用原论文“同构服务器 + 单一负载”的建模思路，将 CPU/GPU 资源折算为等效服务器，用于对比异构感知调度的收益。

`GA`：单目标遗传算法对比优化器。它使用与 NSGA-II 相同的 CPU/GPU 开机数、`defer_ratio` 和 `migration_ratio` 编码，将成本、时延、SLA、负载均衡和约束惩罚加权为单一适应度后搜索调度方案。

`PSO`：单目标粒子群对比优化器。它采用连续粒子位置表示调度变量，评价时按整数编码 round/clip 成可执行方案，用于对比启发式搜索在同一仿真口径下的表现。

`Proposed`：本文主方法，使用 NSGA-II 同时优化每小时 CPU 开机数、GPU 开机数、时间迁移比例 `defer_ratio` 和空间迁移比例 `migration_ratio`。调度目标包括总成本、平均时延/SLA 和资源协同程度。空间迁移通过外部算力池抽象实现，远端执行任务不计入本地 IT 功率，但会产生远端执行成本、网络能耗和迁移时延惩罚。

## 时空迁移建模说明

时间迁移：对可延迟任务，在高电价时段按 `defer_ratio` 延后到后续时段处理，降低峰时本地用电压力。

空间迁移：对 `migratable=True` 的任务，在高电价、本地功率压力较高或本地 CPU/GPU 容量不足时，按 `migration_ratio` 尝试迁移到外部算力池执行，用于刻画可迁移任务向远端算力池转移后的成本、能耗和 SLA 变化。

跨区时延：用于刻画远端执行带来的服务质量损失，是影响迁移决策和 SLA 违约率的重要因素。模型中远端迁移时延由两部分组成：

```text
远端迁移时延 = 基础跨区传输/调度时延 + 任务级迁移惩罚时延
```

其中，`base_cfg["migration"]["migration_delay_hours"]` 表示基础跨区传输/调度时延，`task.migration_delay_penalty` 表示任务自身对跨区迁移的额外时延惩罚。

外部算力池由 `config/base.yaml` 中的 `migration` 参数控制：

```json
"migration": {
  "enable_spatial_migration": true,
  "max_migration_ratio": 0.55,
  "remote_capacity_cpu": 20.0,
  "remote_capacity_gpu": 14.0,
  "migration_delay_hours": 0.05,
  "migration_cost_per_task": 0.03,
  "network_energy_kwh_per_task": 0.005,
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
- `ga_convergence_curve.png`：GA 加权适应度收敛曲线。
- `pso_convergence_curve.png`：PSO 加权适应度收敛曲线。
- `spatial_migration_bar.png`：各算法远端迁移任务数对比。
- `load_shift_curve.png`：原始本地负荷与时空迁移后本地负荷对比。
- `hourly_power_breakdown.csv`：按小时导出的 CPU/GPU IT 功率、制冷功率、固定功率、远端能耗和等效功率。
- `power_breakdown_stack.png`：CPU 功率、GPU 功率、制冷功率和固定功率堆叠图。
- `time_space_ablation.png`：时空迁移消融图，对比完整 Proposed、无空间迁移和无时间迁移。
- `time_space_migration_effect.png`：无迁移、仅时间迁移、时间+空间迁移的成本与时延对比。
- `rolling_vs_static.png`：Rolling-Proposed 与 Static-Proposed 的成本、时延、SLA 和远端迁移任务数对比。
- `ablation_results.png`：消融实验成本与时延对比。
- `sensitivity_results.png`：灵敏度实验成本与 SLA 对比。
- `cross_region_delay_sensitivity.png`：低/中/高跨区时延下远端迁移任务数和 SLA 违约率对比。

扩展实验表格：

- `outputs/ablation_results.csv`：消融实验结果。
- `outputs/rolling_results.csv`：Static-Proposed 与 Rolling-Proposed 总体结果及各滚动窗口结果。
- `outputs/sensitivity_results.csv`：灵敏度实验结果。

## 结果表格中文表头说明

项目输入数据仍使用英文表头，例如 `data/hourly_input.csv`、`data/processed/hourly_profile.csv`、`data/synthetic/tasks.csv` 和 `data/real/tasks.csv`。这些字段会被程序直接读取，不能改成中文，否则会导致加载和仿真逻辑报错。

论文查看和整理用的输出结果 CSV 使用中文表头导出，包括 `results.csv`、`outputs/proposed_pareto.csv`、`outputs/ablation_results.csv`、`outputs/rolling_results.csv`、`outputs/sensitivity_results.csv`、`outputs/hourly_power_breakdown.csv` 和 `outputs/runtime_benchmark.csv`。导出编码为 `utf-8-sig`，Excel 直接打开时中文表头不应乱码。

程序内部的 DataFrame 计算和图表绘制仍使用英文列名。若后续脚本需要继续读取某些结果数据，建议直接复用内部英文 DataFrame，或在读取中文表头 CSV 后自行做反向映射，不要把中文表头结果文件当作项目输入文件。

## 结果字段说明

`results.csv` 和 `outputs/proposed_pareto.csv` 中包含常规指标：

- `total_energy_kwh`：总能耗，包含本地能耗和远端网络能耗。
- `total_cost`：总成本，包含本地电费、远端迁移成本和可选碳成本。
- `avg_delay_hours`：平均任务时延，包含远端迁移时延。
- `sla_violation_rate`：SLA 违约率。
- `avg_cpu_utilization`：CPU 平均利用率。
- `avg_gpu_utilization`：GPU 平均利用率。
- `load_imbalance`：负载不均衡度。
- `peak_valley_gap_kw`：本地功率峰谷差。
- `unit_token_energy_kwh_per_million`：单位百万 token 能耗。
- `unit_token_cost_per_million`：单位百万 token 成本。

新增空间迁移字段：

- `migration_ratio`：NSGA-II 染色体中的每小时空间迁移比例。
- `remote_task_count`：迁移到外部算力池执行的任务数。
- `remote_completion_rate`：可迁移任务中实际远端完成的比例。
- `remote_cost`：远端迁移执行成本。
- `remote_energy_kwh`：迁移网络能耗。
- `migration_delay_hours`：远端任务平均迁移时延，统计时包含基础跨区传输/调度时延和任务级迁移惩罚时延。

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

## 滚动窗口实验

Rolling-Proposed 采用 4 小时滚动窗口，每个窗口只基于当前队列近似任务、未来 4 小时电价和资源约束重新运行一个轻量 Proposed 短视窗优化。窗口内最优调度方案会拼接为 24 小时调度曲线，并用统一的 `simulate_schedule` 口径与 Static-Proposed 对比。

滚动窗口结果会导出到：

```text
outputs/rolling_results.csv
outputs/rolling_vs_static.png
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
- `低跨区时延`：`migration_delay_hours = 0.05`
- `中跨区时延`：`migration_delay_hours = 0.20`
- `高跨区时延`：`migration_delay_hours = 0.50`
- `高电价波动`

### 跨区时延敏感性实验

跨区时延敏感性实验固定其他调度输入，分别设置低/中/高三档基础跨区传输时延，观察远端迁移任务数、平均时延和 SLA 违约率的变化。结果写入 `outputs/sensitivity_results.csv`，并额外生成 `outputs/cross_region_delay_sensitivity.png`。

灵敏度结果会导出到：

```text
outputs/sensitivity_results.csv
outputs/sensitivity_results.png
outputs/cross_region_delay_sensitivity.png
```

## 参数在哪里修改

常用参数集中在 `config/` 目录：

- 修改 CPU/GPU 服务器数量、功率、服务率：`config/resource.yaml`
- 修改 SLA、功率上限、峰时削减、空间迁移参数：`config/base.yaml`
- 修改 NSGA-II 种群规模、GA/PSO 默认规模、迭代代数、输出路径：`config/experiment.yaml`
- 修改电价输入路径和高低电价阈值：`config/price.yaml`

当前 `.yaml` 文件使用 JSON 兼容写法，因此可以用普通文本编辑器直接修改。

## 大规模仿真数据与运行时间测试

默认主实验仍使用 `config/experiment.yaml` 中的 `paths.tasks_path`，也就是 `data/synthetic/tasks.csv`。该文件当前是约 4474 个任务的小规模数据，用于保证论文主实验稳定、快速、可复现。下面的大规模数据生成和 benchmark 脚本只用于扩展实验或运行时间压力测试，不会改变 `main.py` 的默认行为。

生成 10 倍负载任务数据：

```powershell
python scripts/generate_large_synthetic.py --scale 10 --output data/synthetic/tasks_10x.csv
```

生成指定任务数量的数据：

```powershell
python scripts/generate_large_synthetic.py --target-tasks 100000 --output data/synthetic/tasks_100000.csv
```

生成接近 64MB 的任务 CSV：

```powershell
python scripts/generate_large_synthetic.py --target-mb 64 --output data/synthetic/tasks_64mb.csv
```

生成脚本会读取现有 24 小时负载、电价和碳因子曲线，复用 `models.task.generate_synthetic_tasks` 的字段结构，输出与 `models.task.load_tasks_csv` 兼容的标准任务 CSV。生成报告写入：

```text
outputs/large_synthetic_report.txt
```

报告包含任务总数、CSV 文件大小、三类任务数量、CPU/GPU 任务数量、各小时任务分布，以及平均 CPU/GPU/内存需求。

测试单轮仿真运行时间建议先用 quick 模式：

```powershell
python scripts/benchmark_runtime.py --tasks-path data/synthetic/tasks_10x.csv --mode quick --repeat 1
python scripts/benchmark_runtime.py --tasks-path data/synthetic/tasks_64mb.csv --mode quick --repeat 1
```

quick 模式只计时核心步骤：数据加载、FCFS schedule 构建、FCFS 单次仿真、Price-Only schedule 构建、Price-Only 单次仿真，以及一个简化 Proposed 调度方案的单次仿真。它不会默认运行完整 NSGA-II、GA、PSO、消融实验、灵敏度实验或滚动窗口实验。

benchmark 结果默认写入：

```text
outputs/runtime_benchmark.csv
```

输出字段包括 `scale_or_tasks_path`、`task_count`、`csv_size_mb`、`mode`、`step_name`、`elapsed_seconds` 和 `repeat_index`。

full 模式会调用完整 `experiments.exp_main.run_main_experiment()`，会重复运行 NSGA-II、GA、PSO、消融、滚动窗口和灵敏度实验，并覆盖常规实验输出。大规模数据尤其是接近 64MB 的 CSV 不建议直接用于 full 模式，建议先用 quick benchmark 判断单轮仿真耗时，再决定是否把大规模数据接入主实验。

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
