# 论文建议图表清单

## 优先使用图

1. 电价与功率上限关系：`outputs/nbsdc_fusion/price_powercap.png`
2. 服务器级24小时任务到达量：`outputs/nbsdc_fusion/hourly_task_arrivals.png`
3. DVFS频率与实际功率关系：`outputs/nbsdc_fusion/dvfs_frequency_power.png`
4. 三层数据融合下的等效功率裕度：`outputs/nbsdc_fusion/three_layer_power_margin.png`
5. 基准裕度与三层融合裕度对比：`outputs/nbsdc_fusion/power_margin_baseline_vs_fused.png`
6. 不同机房小时级任务分布热力图：`outputs/nbsdc_fusion/room_hourly_task_heatmap.png`
7. 等效功率裕度与Token产出能力时序关系：`outputs/token_export/power_margin_token_capacity_timeseries.png`
8. 不同Token出口策略净收益对比：`outputs/token_export/token_profit_comparison.png`
9. 不同地区Token出口量对比：`outputs/token_export/token_export_by_region.png`
10. 不同策略跨时区服务时延对比：`outputs/token_export/cross_timezone_latency.png`
11. Token出口关键参数敏感性分析：`outputs/token_export/token_sensitivity.png`

## 辅助图

- 小时级Token产出能力：`outputs/token_export/hourly_token_capacity.png`
- RH-TEO策略下跨时区Token分配曲线：`outputs/token_export/rh_teo_allocation_curve.png`
- 芯片实际功率与功率上限对比：`outputs/nbsdc_fusion/chip_actual_vs_cap.png`
- 不同机房任务总量分布（辅助）：`outputs/nbsdc_fusion/room_task_distribution.png`
- 等效功率裕度到Token产出的转换关系：`outputs/token_export/power_to_token_curve.png`，该图天然近似线性，只建议作为模型关系说明，不建议作为核心实验图。

## 表

- 三个 NBSDC 数据集字段用途说明：README 数据集使用方式表。
- 三层融合小时级指标：`outputs/nbsdc_fusion/aligned_hourly_fusion.csv`
- 三层融合指标摘要：`outputs/nbsdc_fusion/nbsdc_fusion_metrics.csv`
- 机房任务分布：`outputs/nbsdc_fusion/room_task_distribution.csv`
- Token 出口区域配置：`config/token_export.yaml`
- Token 出口策略对比结果：`outputs/token_export/token_export_results.csv`
- Token 出口策略对比中文展示版：`outputs/token_export/token_export_results_cn.csv`，包含“不出口”零出口基准。
- 参数敏感性结果：`outputs/token_export/token_sensitivity_results.csv`

说明：主要图表已排除“不出口”基准，以突出实际发生 Token 出口的策略之间的差异；“不出口”仅保留在结果表中用于对照。

## `token_sensitivity.png` 读图说明

- 蓝线表示净收益。
- 红线表示 SLA 违约率。
- 绿线表示平均跨时区时延。
- 橙线表示北美出口占比。
- 北美被选作高价格、长时延远端出口市场代表，用于观察远端市场价格变化对 Token 出口分配和净收益的影响。
