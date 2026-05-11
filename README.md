# 异构资源环境下数据中心算电协同调度优化研究

当前项目主线为：**基于 NBSDC 多层级数据融合的算电协同调度与 Token 出口优化研究**。

项目综合利用 NBSDC 的集群级功率封顶数据、服务器级任务调度数据和芯片级 DVFS 数据，构建“电力约束-任务负载-设备功率响应”的三层融合模型。在此基础上，将等效功率裕度折算为 AI 推理 Token 产出能力，并实现基于滚动时域的跨时区 Token 出口优化策略 RH-TEO，用于分析电力成本、Token 收益、服务价格和跨时区时延之间的权衡关系。

## 数据集使用方式

| 数据文件 | 使用层级 | 主要字段 | 在项目中的作用 |
| --- | --- | --- | --- |
| `data/raw/数据中心集群级别的调度数据.xlsx` | 集群级 | 电价、功率上限、奖励值 | 构建分时电价、Power Capping 和集群功率约束 |
| `data/raw/服务器级别的调度数据.xlsx` | 服务器级 | 任务到达、CPU 需求、deadline、机房/机架/服务器分布 | 构建 24 小时任务负载、层级负载分布和 deadline 违约统计 |
| `data/raw/芯片级别的调度数据.xlsx` | 芯片级 | DVFS 频率、实际功率、报量功率、芯片功率上限 | 构建设备功率响应与 DVFS 跟踪误差 |

## 运行流程

```powershell
python scripts/prepare_uploaded_case_dataset.py
python run_nbsdc_fusion.py
python run_token_export.py
```

清洗脚本支持服务器任务时间映射模式：

```powershell
python scripts/prepare_uploaded_case_dataset.py --arrival-time-mode auto
python scripts/prepare_uploaded_case_dataset.py --arrival-time-mode rescale_24h
python scripts/prepare_uploaded_case_dataset.py --arrival-time-mode raw_step
```

默认 `auto` 会检查 `arrival_step` 范围和任务小时分布；当原始步长不足 288 或任务集中在少数小时内时，自动采用 `rescale_24h` 把任务映射到 0-23 小时。

## 输出目录

`data/real_case/` 保存清洗后的程序输入 CSV：

- `cluster_power_price_5min.csv`
- `server_tasks_5min_raw_mapped.csv`
- `server_tasks_24h.csv`
- `hourly_input_24h.csv`
- `chip_dvfs.csv`

规范输出按类型拆分：

- `outputs/data/nbsdc_fusion/`：NBSDC 三层融合 CSV 表格。
- `outputs/data/token_export/`：Token 出口实验 CSV 表格。
- `outputs/figures/nbsdc_fusion/`：NBSDC 三层融合 PNG/PDF 图。
- `outputs/figures/token_export/`：Token 出口实验 PNG/PDF 图。
- `outputs/reports/`：清洗报告和实验摘要。

为兼容旧引用，运行脚本后仍会在 `outputs/nbsdc_fusion/` 和 `outputs/token_export/` 保留一份同名副本。

`outputs/data/nbsdc_fusion/` 保存三层融合表格：

- `aligned_hourly_fusion.csv`
- `aligned_hourly_fusion_cn.csv`
- `nbsdc_fusion_metrics.csv`
- `nbsdc_fusion_metrics_cn.csv`
- `room_task_distribution.csv`
- `room_task_distribution_cn.csv`
- `rack_task_distribution.csv`
- `rack_task_distribution_cn.csv`
- `server_task_distribution.csv`
- `server_task_distribution_cn.csv`

`outputs/figures/nbsdc_fusion/` 保存三层融合图：

- `price_powercap.png`
- `hourly_task_arrivals.png`
- `dvfs_frequency_power.png`
- `three_layer_power_margin.png`
- `power_margin_baseline_vs_fused.png`
- `room_hourly_task_heatmap.png`
- `room_task_distribution.png`
- `chip_actual_vs_cap.png`

`outputs/data/token_export/` 保存 Token 出口实验表格：

- `token_export_results.csv`
- `token_export_results_cn.csv`
- `hourly_token_export.csv`
- `hourly_token_export_cn.csv`
- `region_token_export.csv`
- `region_token_export_cn.csv`
- `token_sensitivity_results.csv`
- `token_sensitivity_results_cn.csv`

`outputs/figures/token_export/` 保存 Token 出口实验图：

- `hourly_token_capacity.png`
- `power_margin_token_capacity_timeseries.png`
- `token_export_by_region.png`
- `token_profit_comparison.png`
- `cross_timezone_latency.png`
- `rh_teo_allocation_curve.png`
- `token_sensitivity.png`
- `power_to_token_curve.png`，该图仅作为辅助检查，不建议作为核心论文图。

`outputs/reports/` 保存摘要和报告：

- `uploaded_case_cleaning_report.txt`
- `nbsdc_fusion_summary.txt`
- `token_export_summary.txt`

说明：`token_export_results.csv` 和 `token_export_results_cn.csv` 保留“不出口”零出口基准；主要柱状图和地区出口图已排除该基准，以突出实际发生 Token 出口的策略差异。

项目同时输出英文标准版 CSV 与中文展示版 CSV。英文版用于程序复现和后续读取，中文 `*_cn.csv` 用于论文表格整理；中文展示版会翻译表头以及策略名、地区名、参数名等字段值。

## 三层融合模型

当前融合模型输出以下关键指标：

- `cluster_cap_norm`：集群功率上限。如果原始功率上限已经在 0-1 之间，直接裁剪到 0-1；否则使用 min-max 归一化。
- `server_load_norm`：服务器负载，使用小时 CPU 需求的 min-max 归一化。
- `chip_power_variation_norm`：芯片实际功率相对波动，使用小时实际功率的 min-max 归一化。
- `chip_power_ratio`：芯片实际功率与芯片功率上限的比例，只作为辅助指标，不直接作为裕度扣减项。

功率裕度公式为：

```text
power_margin_norm =
cluster_cap_norm
- 0.55 * server_load_norm
- 0.25 * chip_power_variation_norm
- 0.10
```

由于芯片实际功率存在较高基础功耗，项目采用 min-max 归一化刻画芯片功率的相对波动，避免最大值归一化导致曲线过平。`aligned_hourly_fusion.csv` 同时保留 `power_margin_norm_old`，便于对比旧公式与新公式。

## Token 出口优化

RH-TEO 表示 Rolling-Horizon Token Export Optimization，主流程为：

1. 对集群级功率上限、服务器级负载、芯片级实际功率进行小时级对齐。
2. 计算等效功率裕度。
3. 将功率裕度折算为 Token 产出能力：

```text
token_capacity_t = power_margin_norm_t * token_per_margin_unit
```

4. 对国内、欧洲、北美三个 Token 服务场景进行滚动窗口分配，综合收益、电力成本、时延惩罚、SLA 惩罚和出口波动惩罚选择当前小时决策。

本项目以中国数据中心为算力供给侧，构造国内本地需求、欧洲出口需求和北美出口需求三类 Token 服务场景。欧洲和北美区域参数为扩展场景假设，用于刻画跨境算力服务的价格与时延差异，并非 NBSDC 原始字段。

价格敏感性实验选取北美作为高价格、长时延远端出口市场代表，用于刻画远端市场价格变化对 Token 出口分配和净收益的影响。

项目实现了五类策略对比：

- `No-Export`
- `Power-Margin-Only`
- `Price-Driven`
- `Latency-Aware`
- `RH-TEO`

## 重要声明

NBSDC 数据集本身不直接提供线上请求级 Token 字段。本项目的 Token 出口实验是基于 NBSDC 三层数据融合得到的等效功率裕度构建的扩展场景，用于近似刻画“电力-算力-Token”转换链路。

因此，代码不会将 NBSDC 原始任务强行改造成 CPU/GPU/token_batch/remote_pool 任务，也不会把 Token 出口结果表作为原始真实请求数据解释。

## 国内视角场景设定说明

本项目以中国数据中心为算力供给侧，构造国内、欧洲、北美三类 Token 服务场景。欧洲和北美参数为扩展场景假设，不是 NBSDC 原始字段。

| 场景 | 定位 | 时延特征 | 价格特征 |
| --- | --- | --- | --- |
| 国内 | 本地/近域需求 | 低时延 | 低价格 |
| 欧洲 | 中距离跨境出口 | 中等时延 | 中等价格 |
| 北美 | 远端高价出口市场 | 高时延 | 高价格 |

价格敏感性实验选取北美作为高价格、长时延远端出口市场代表，用于刻画远端市场价格变化对 Token 出口分配和净收益的影响。

## 模型假设与局限性

- NBSDC 不包含真实 Token 请求。
- Token 出口为基于三层数据融合结果构造的扩展场景。
- 等效功率裕度是归一化指标，不等同于真实剩余功率。
- 国内、欧洲、北美参数用于刻画价格和时延差异，不是 NBSDC 原始字段。

## 旧项目归档

原 CPU/GPU synthetic 优化实验已归档到：

```text
archive/legacy_synthetic_optimization/
```

归档内容包括旧的 `models/`、`schedulers/`、`optimizers/`、`main.py`、NSGA-II/GA/PSO 对比、消融实验、滚动实验、灵敏度实验和合成任务数据生成逻辑。该部分保留用于追溯，不作为当前论文主线。
