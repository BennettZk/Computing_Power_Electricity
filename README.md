# 异构资源环境下数据中心算电协同调度优化

本项目是在原“面向电价型需求响应的数据中心能耗多目标联合优化策略”复现代码基础上做的增量式改造。改造目标是保留原论文中的分时电价、能耗成本、任务时延、需求响应和多目标优化主线，同时把“同构服务器 + 单一负载”扩展为“CPU/GPU 异构资源池 + 多类型任务”的单数据中心调度实验。

项目当前聚焦本科毕设可解释、可运行、可出图的最小可行版本，不涉及区块链、真实 token 交易、跨区域云边协同等复杂扩展。

## 一、项目能解决什么问题

给定 24 小时内的任务到达量、分时电价、碳因子和数据中心 CPU/GPU 资源配置，项目会比较不同调度策略在以下指标上的表现：

- 总能耗
- 总电费
- 平均任务时延
- SLA 违约率
- CPU 利用率
- GPU 利用率
- 负载不均衡度
- 峰谷差
- 单位 token 电耗
- 单位 token 电费

核心问题可以概括为：

在电价随时间变化、任务类型不同、CPU/GPU 资源能力不同的情况下，如何决定每个小时开启多少 CPU/GPU 服务器，以及是否延后可延迟任务，从而在电费、时延和资源均衡之间取得折中。

## 二、目录结构

```text
dc_token_opt/
├─ config/
│  ├─ base.yaml              # 基础参数、功率上限、SLA 约束、惩罚参数
│  ├─ price.yaml             # 分时电价数据路径和高低电价分位数
│  ├─ resource.yaml          # CPU/GPU 异构资源池参数
│  └─ experiment.yaml        # 随机种子、输出路径、NSGA-II 参数
├─ data/
│  ├─ hourly_input.csv       # 原始小时级负载、电价、碳因子输入
│  ├─ processed/             # 自动生成的处理后小时曲线
│  └─ synthetic/             # 自动生成的任务序列
├─ models/
│  ├─ task.py                # 任务模型与合成任务生成
│  ├─ resource.py            # CPU/GPU 资源模型
│  ├─ power_model.py         # 同构/异构功率与成本计算
│  ├─ delay_model.py         # M/M/c 时延近似模型
│  └─ objective.py           # 调度仿真、指标汇总、约束惩罚
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
│  ├─ exp_ablation.py        # 消融实验预留
│  └─ exp_sensitivity.py     # 灵敏度实验预留
├─ utils/
│  ├─ io_utils.py            # 配置、数据和输出目录工具
│  ├─ metrics.py             # 指标计算
│  ├─ plotting.py            # 图表绘制
│  └─ seed.py                # 随机种子设置
├─ main.py                   # 一键运行入口
├─ results.csv               # 主实验结果表
├─ summary.txt               # 实验摘要
└─ outputs/                  # 自动生成图表和帕累托前沿
```

## 三、运行环境

推荐直接使用项目已有虚拟环境：

```powershell
.\.venv\Scripts\python.exe main.py
```

如果需要重新安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖包在 `requirements.txt` 中：

```text
numpy
pandas
matplotlib
scipy
pymoo
```

## 四、怎么运行

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

两种命令都会执行同一套主流程。

## 五、输入数据格式

### 1. 小时级输入数据

默认读取：

```text
data/hourly_input.csv
```

字段说明：

- `hour`：小时编号，范围为 0 到 23
- `arrival_rate`：该小时任务到达量
- `price`：该小时电价
- `carbon_factor`：该小时碳排因子

首次运行时，程序会生成处理后的文件：

```text
data/processed/hourly_profile.csv
```

该文件会额外包含：

- `renewable_ratio`：绿电占比近似参数

### 2. 任务序列输入

默认任务文件：

```text
data/synthetic/tasks.csv
```

如果文件不存在，程序会根据小时级负载自动生成合成任务。

字段说明：

- `task_id`：任务编号
- `arrival_time`：任务到达时隙
- `task_type`：任务类型，包含 `delay_sensitive`、`delay_tolerant`、`token_batch`
- `cpu_demand`：CPU 计算需求
- `gpu_demand`：GPU 计算需求
- `memory_demand`：内存需求
- `bandwidth_demand`：带宽需求
- `token_amount`：token 工作量，非 token 任务可为 0
- `deadline`：任务截止时隙
- `priority`：任务优先级，数值越大优先级越高

## 六、四种算法说明

### 1. FCFS

先来先服务基线。它根据每小时任务数量估算 CPU/GPU 开机台数，不考虑分时电价，也不主动延迟任务。

### 2. Price-Only

仅电价响应基线。高电价时段减少可延迟任务对应的开机规模，低电价时段适当增加开机数释放任务，但不使用 NSGA-II 进行联合优化。

### 3. Homogeneous-Baseline

同构服务器基线。它保留原论文“单一服务器类型 + 单一负载”的建模思路，将 CPU/GPU 参数折算成等效服务器，作为异构调度的对照组。

### 4. Proposed

本文方法。它使用 NSGA-II 同时优化每小时 CPU 开机数、GPU 开机数和可延迟任务比例，在总电费、时延/SLA 和资源负载均衡之间寻找帕累托折中解。

## 七、输出结果

主实验运行后会生成：

```text
results.csv
summary.txt
outputs/proposed_pareto.csv
```

图表输出在：

```text
outputs/
```

包含：

- `price_load_curve.png`：电价-负荷曲线
- `pareto_front.png`：Proposed 时延-成本帕累托前沿
- `energy_bar.png`：各算法总能耗柱状图
- `cpu_gpu_utilization.png`：CPU/GPU 利用率对比图
- `convergence_curve.png`：NSGA-II 收敛曲线

## 八、参数在哪里改

常用参数集中在 `config/` 目录：

- 修改 CPU/GPU 服务器数量、功率、服务率：`config/resource.yaml`
- 修改最大平均时延、功率上限、峰时削减比例：`config/base.yaml`
- 修改 NSGA-II 种群规模、迭代代数、输出路径：`config/experiment.yaml`
- 修改电价输入路径和高低电价阈值分位数：`config/price.yaml`

注意：当前 `.yaml` 文件使用 JSON 兼容格式写法，因此可以用普通文本编辑器直接修改。

## 九、论文写作时可以怎么解释模型简化

当前版本做了以下合理简化：

- 单数据中心场景，不考虑跨区域调度。
- 任务不可拆分，一个任务只进入 CPU 或 GPU 中的一个资源队列。
- CPU/GPU 队列分别使用 M/M/c 近似估计时延。
- token 只作为 AI 类任务工作量指标，不涉及区块链或真实交易。
- 绿电占比作为电费折扣近似，碳成本参数默认关闭。
- 制冷功率用 CPU/GPU IT 功率乘以资源类型制冷系数近似。

这些简化的目的不是构建工业级仿真器，而是保证实验逻辑清晰、结果可复现、图表可用于本科论文。

## 十、常见问题

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
