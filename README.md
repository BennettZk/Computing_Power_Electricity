# 异构资源环境下数据中心算电协同调度优化研究

本项目当前主线为：**基于 NBSDC 多层级数据融合的算电协同调度与 Token 出口优化研究**。

项目综合利用 NBSDC 的集群级功率封顶数据、服务器级任务调度数据和芯片级 DVFS 数据，构建“电力约束-任务负载-设备功率响应”的三层融合模型。在此基础上，将等效功率裕度折算为 AI 推理 Token 产出能力，并实现基于滚动时域的跨时区 Token 出口优化策略 RH-TEO，用于分析电力成本、Token 收益、服务价格和跨时区时延之间的权衡关系。

## 数据集使用方式

| 数据文件 | 使用层级 | 主要字段 | 在项目中的作用 |
| --- | --- | --- | --- |
| `data/raw/数据中心集群级别的调度数据.xlsx` | 集群级 | 电价、功率上限、奖励值 | 构建分时电价、Power Capping 和集群功率约束 |
| `data/raw/服务器级别的调度数据.xlsx` | 服务器级 | 任务到达、CPU 需求、deadline、机房/机架/服务器分布 | 构建 24 小时任务负载、层级负载分布和 deadline 违约统计 |
| `data/raw/芯片级别的调度数据.xlsx` | 芯片级 | DVFS 频率、实际功率、报量功率、芯片功率上限 | 构建设备功率响应与 DVFS 跟踪误差 |

## 运行流程

在项目根目录依次运行：

```powershell
python scripts/prepare_uploaded_case_dataset.py
python run_nbsdc_fusion.py
python run_token_export.py
```

如果使用虚拟环境中的解释器：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_uploaded_case_dataset.py
.\.venv\Scripts\python.exe run_nbsdc_fusion.py
.\.venv\Scripts\python.exe run_token_export.py
```

## 输出目录

`data/real_case/` 保存清洗后的程序输入 CSV：

- `cluster_power_price_5min.csv`
- `server_tasks_5min_raw_mapped.csv`
- `server_tasks_24h.csv`
- `hourly_input_24h.csv`
- `chip_dvfs.csv`

`outputs/nbsdc_fusion/` 保存三层融合结果：

- `aligned_hourly_fusion.csv`
- `nbsdc_fusion_metrics.csv`
- `nbsdc_fusion_summary.txt`
- `price_powercap.png`
- `hourly_task_arrivals.png`
- `dvfs_frequency_power.png`
- `three_layer_power_margin.png`
- `room_task_distribution.png`
- `chip_actual_vs_cap.png`

`outputs/token_export/` 保存 Token 出口实验结果：

- `token_export_results.csv`
- `token_export_results_cn.csv`
- `hourly_token_export.csv`
- `region_token_export.csv`
- `token_sensitivity_results.csv`
- `token_export_summary.txt`
- `hourly_token_capacity.png`
- `token_export_by_region.png`
- `token_profit_comparison.png`
- `cross_timezone_latency.png`
- `power_to_token_curve.png`
- `rh_teo_allocation_curve.png`
- `token_sensitivity.png`

## 算法说明

RH-TEO 表示 Rolling-Horizon Token Export Optimization，主流程为：

1. 对集群级功率上限、服务器级负载、芯片级实际功率进行小时级对齐。
2. 将三类指标归一化到 0-1 的等效功率尺度。
3. 计算等效功率裕度：

```text
power_margin_norm = cluster_cap_norm - alpha * server_load_norm - beta * chip_power_norm
```

4. 将功率裕度折算为 Token 产出能力：

```text
token_capacity_t = power_margin_norm_t * token_per_margin_unit
```

5. 对 Asia、Europe、America 三个跨时区出口区域进行滚动窗口分配，综合收益、电力成本、时延惩罚、SLA 惩罚和出口波动惩罚选择当前小时决策。

项目实现了五类策略对比：

- `No-Export`
- `Power-Margin-Only`
- `Price-Driven`
- `Latency-Aware`
- `RH-TEO`

## 重要声明

NBSDC 数据集本身不直接提供线上请求级 Token 字段。本项目的 Token 出口实验是基于 NBSDC 三层数据融合得到的等效功率裕度构建的扩展场景，用于近似刻画“电力-算力-Token”转换链路。

因此，代码不会将 NBSDC 原始任务强行改造成 CPU/GPU/token_batch/remote_pool 任务，也不会把 Token 出口结果表作为原始真实请求数据解释。

## 旧项目归档

原 CPU/GPU synthetic 优化实验已归档到：

```text
archive/legacy_synthetic_optimization/
```

归档内容包括旧的 `models/`、`schedulers/`、`optimizers/`、`main.py`、NSGA-II/GA/PSO 对比、消融实验、滚动实验、灵敏度实验和合成任务数据生成逻辑。该部分保留用于追溯，不作为当前论文主线。
