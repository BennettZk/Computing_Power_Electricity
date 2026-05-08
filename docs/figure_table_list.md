# 论文建议图表清单

## 优先使用图

1. 电价与功率上限关系：`outputs/nbsdc_fusion/price_powercap.png`
2. 服务器级24小时任务到达量：`outputs/nbsdc_fusion/hourly_task_arrivals.png`
3. DVFS频率与实际功率关系：`outputs/nbsdc_fusion/dvfs_frequency_power.png`
4. 三层数据融合下的等效功率裕度：`outputs/nbsdc_fusion/three_layer_power_margin.png`
5. 新旧等效功率裕度对比：`outputs/nbsdc_fusion/power_margin_old_vs_new.png`
6. 等效功率裕度与Token产出能力时序关系：`outputs/token_export/power_margin_token_capacity_timeseries.png`
7. 不同Token出口策略净收益对比：`outputs/token_export/token_profit_comparison.png`
8. 不同地区Token出口量对比：`outputs/token_export/token_export_by_region.png`
9. 不同策略跨时区服务时延对比：`outputs/token_export/cross_timezone_latency.png`
10. Token出口关键参数敏感性分析：`outputs/token_export/token_sensitivity.png`

## 辅助图

- 小时级Token产出能力：`outputs/token_export/hourly_token_capacity.png`
- RH-TEO策略下跨时区Token分配曲线：`outputs/token_export/rh_teo_allocation_curve.png`
- 芯片实际功率与功率上限对比：`outputs/nbsdc_fusion/chip_actual_vs_cap.png`
- 不同机房任务分布：`outputs/nbsdc_fusion/room_task_distribution.png`
- 等效功率裕度到Token产出的转换关系：`outputs/token_export/power_to_token_curve.png`，该图天然近似线性，只建议作为模型关系说明，不建议作为核心实验图。

## 表

- 三个 NBSDC 数据集字段用途说明：README 数据集使用方式表。
- 三层融合小时级指标：`outputs/nbsdc_fusion/aligned_hourly_fusion.csv`
- 三层融合指标摘要：`outputs/nbsdc_fusion/nbsdc_fusion_metrics.csv`
- 机房任务分布：`outputs/nbsdc_fusion/room_task_distribution.csv`
- Token 出口区域配置：`config/token_export.yaml`
- Token 出口策略对比结果：`outputs/token_export/token_export_results.csv`
- Token 出口策略对比中文展示版：`outputs/token_export/token_export_results_cn.csv`
- 参数敏感性结果：`outputs/token_export/token_sensitivity_results.csv`
