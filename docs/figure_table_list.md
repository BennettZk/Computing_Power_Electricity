# 论文建议图表清单

本文档列出运行 `main.py` 后生成的主要图表和结果表。所有 `outputs/` 内容均为本地生成产物，默认不上传 GitHub。

## 推荐运行方式

```powershell
python main.py
```

运行前只需要把三份原始 Excel 放入 `data/raw/`。

## 核心图

| 序号 | 图 | 路径 | 用途 |
| --- | --- | --- | --- |
| 1 | 电价与功率上限关系 | `outputs/figures/nbsdc_fusion/price_powercap.png` | 展示集群级电价与功率约束 |
| 2 | 服务器级 24 小时任务到达量 | `outputs/figures/nbsdc_fusion/hourly_task_arrivals.png` | 展示业务负载的小时分布 |
| 3 | DVFS 频率下实际功率分布 | `outputs/figures/nbsdc_fusion/dvfs_frequency_power.png` | 展示芯片级功率响应 |
| 4 | 三层数据融合下的等效功率裕度 | `outputs/figures/nbsdc_fusion/three_layer_power_margin.png` | 展示核心融合指标 |
| 5 | 基准裕度与三层融合裕度对比 | `outputs/figures/nbsdc_fusion/power_margin_baseline_vs_fused.png` | 说明新旧裕度公式差异 |
| 6 | 不同机房小时级任务分布热力图 | `outputs/figures/nbsdc_fusion/room_hourly_task_heatmap.png` | 展示空间和时间负载分布 |
| 7 | 功率裕度与 Token 产出能力时序关系 | `outputs/figures/token_export/power_margin_token_capacity_timeseries.png` | 说明裕度到 Token 产能的映射 |
| 8 | Token 出口策略收益与时延权衡 | `outputs/figures/token_export/token_profit_latency_tradeoff.png` | 核心策略对比图 |
| 9 | 不同 Token 出口策略净收益对比 | `outputs/figures/token_export/token_profit_comparison.png` | 展示收益差异 |
| 10 | 不同地区 Token 出口量对比 | `outputs/figures/token_export/token_export_by_region.png` | 展示区域分配结果 |
| 11 | Token 出口关键参数敏感性分析 | `outputs/figures/token_export/token_sensitivity.png` | 展示 SLA、时延、价格变化影响 |

## 辅助图

| 图 | 路径 | 说明 |
| --- | --- | --- |
| 跨时区服务时延对比 | `outputs/figures/token_export/cross_timezone_latency.png` | 可作为策略时延对比补充 |
| 小时级 Token 产出能力 | `outputs/figures/token_export/hourly_token_capacity.png` | 可说明 Token capacity 的小时变化 |
| RH-TEO 区域分配曲线 | `outputs/figures/token_export/rh_teo_allocation_curve.png` | 可说明滚动策略的区域选择 |
| 芯片实际功率与功率上限对比 | `outputs/figures/nbsdc_fusion/chip_actual_vs_cap.png` | 可作为芯片功率响应补充 |
| 不同机房任务总量分布 | `outputs/figures/nbsdc_fusion/room_task_distribution.png` | 可作为空间负载分布补充 |
| 功率裕度到 Token 产出的转换关系 | `outputs/figures/token_export/power_to_token_curve.png` | 建议作为模型关系说明图 |

## 结果表

| 表 | 路径 | 说明 |
| --- | --- | --- |
| 三层融合小时级指标 | `outputs/data/nbsdc_fusion/aligned_hourly_fusion.csv` | 核心融合结果 |
| 三层融合指标摘要 | `outputs/data/nbsdc_fusion/nbsdc_fusion_metrics.csv` | 论文分析常用字段 |
| 机房任务分布 | `outputs/data/nbsdc_fusion/room_task_distribution.csv` | 任务空间分布 |
| 机架任务分布 | `outputs/data/nbsdc_fusion/rack_task_distribution.csv` | 任务空间分布 |
| 服务器任务分布 | `outputs/data/nbsdc_fusion/server_task_distribution.csv` | 任务空间分布 |
| Token 出口策略对比 | `outputs/data/token_export/token_export_results.csv` | 核心策略结果 |
| Token 出口策略对比中文展示版 | `outputs/data/token_export/token_export_results_cn.csv` | 论文表格整理用 |
| 小时级 Token 出口结果 | `outputs/data/token_export/hourly_token_export.csv` | 分小时策略分析 |
| 区域 Token 出口结果 | `outputs/data/token_export/region_token_export.csv` | 区域分配分析 |
| 参数敏感性结果 | `outputs/data/token_export/token_sensitivity_results.csv` | 敏感性分析 |

## 报告文件

| 报告 | 路径 |
| --- | --- |
| 原始数据清洗报告 | `outputs/reports/uploaded_case_cleaning_report.txt` |
| NBSDC 融合实验摘要 | `outputs/reports/nbsdc_fusion_summary.txt` |
| Token 出口优化实验摘要 | `outputs/reports/token_export_summary.txt` |

## 使用说明

主要图表已经排除了 `No-Export` 零出口基准，以突出实际发生 Token 出口的策略差异。`No-Export` 仍保留在结果表中，用作零出口对照。

`outputs/figures_final/` 会保存一份图表副本，便于论文排版时集中取图。若重新运行 `main.py`，该目录中的同名图表会随运行结果更新。
