# 面向跨域算力服务的数据中心算电协同裕度分配

本项目以 NBSDC 的三类原始调度数据为输入，完成从数据清洗、三层数据融合到 Token 出口优化实验的完整流程。当前主线是：

1. 使用集群级、服务器级、芯片级数据构建小时级融合指标。
2. 计算等效功率裕度 `power_margin_norm`。
3. 将功率裕度折算为 Token 产出能力。
4. 使用 RH-TEO 滚动时域策略进行跨时区 Token 出口分配。

## 快速运行

把三份原始 Excel 文件放到 `data/raw/` 下，然后在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

如果已经激活虚拟环境，也可以运行：

```powershell
python main.py
```

`main.py` 会自动完成：

- 识别 `data/raw/` 下的集群级、服务器级、芯片级 Excel 文件
- 清洗并生成 `data/real_case/` 中间 CSV
- 运行 NBSDC 三层融合实验
- 运行 Token 出口优化实验
- 将结果写入 `outputs/`

可选参数：

```powershell
python main.py --arrival-time-mode auto
python main.py --arrival-time-mode rescale_24h
python main.py --arrival-time-mode raw_step
python main.py --raw-dir data/raw
```

默认 `auto` 会根据服务器任务的 `arrival_step` 范围和小时分布自动选择映射方式。

## 输入数据

`data/raw/` 是唯一需要手动放入数据的目录。建议保留原始中文文件名，也可以使用包含以下关键词的英文文件名：

| 数据层级 | 识别关键词 | 主要用途 |
| --- | --- | --- |
| 集群级 | `cluster`、`datacenter`、`data_center`、`数据中心`、`集群` | 电价、功率上限、Power Capping 场景 |
| 服务器级 | `server`、`服务器` | 任务到达、CPU 需求、deadline、机房/机架/服务器分布 |
| 芯片级 | `chip`、`dvfs`、`芯片` | DVFS 频率、实际功率、报量功率、芯片功率上限 |

如果文件名无法判断，程序会根据工作表名称辅助识别服务器级和芯片级文件。

## 输出目录

运行结果会生成在以下目录：

| 目录 | 内容 |
| --- | --- |
| `data/real_case/` | 清洗后的程序输入 CSV |
| `outputs/data/nbsdc_fusion/` | NBSDC 三层融合 CSV |
| `outputs/data/token_export/` | Token 出口优化 CSV |
| `outputs/figures/nbsdc_fusion/` | NBSDC 融合图表 |
| `outputs/figures/token_export/` | Token 出口优化图表 |
| `outputs/figures_final/` | 论文整理用图表副本 |
| `outputs/reports/` | 清洗报告和实验摘要 |
| `outputs/nbsdc_fusion/`、`outputs/token_export/` | 兼容旧路径的结果副本 |

重要输出文件包括：

- `outputs/data/nbsdc_fusion/aligned_hourly_fusion.csv`
- `outputs/data/nbsdc_fusion/nbsdc_fusion_metrics.csv`
- `outputs/data/token_export/token_export_results.csv`
- `outputs/data/token_export/hourly_token_export.csv`
- `outputs/data/token_export/token_sensitivity_results.csv`
- `outputs/reports/uploaded_case_cleaning_report.txt`
- `outputs/reports/nbsdc_fusion_summary.txt`
- `outputs/reports/token_export_summary.txt`

## Git 提交策略

`data/` 和 `outputs/` 中的数据、图表、报告都是本地运行产物，默认不上传。仓库只保留目录占位文件：

- `data/.gitkeep`
- `data/raw/.gitkeep`
- `outputs/.gitkeep`
- `legacy_workflows/legacy_synthetic_optimization/data/.gitkeep`
- `legacy_workflows/legacy_synthetic_optimization/outputs/.gitkeep`

因此，提交代码时不需要提交原始数据和运行结果。新的原始 Excel 放入 `data/raw/` 后可以直接运行 `main.py` 复现结果；旧流程生成的结果也只保留在本地归档输出目录中。

## 旧入口

以下脚本仍保留用于分步调试：

```powershell
python scripts/prepare_uploaded_case_dataset.py
python run_nbsdc_fusion.py
python run_token_export.py
```

日常使用建议直接运行 `main.py`。

旧版 synthetic 优化方法保留在 `legacy_workflows/legacy_synthetic_optimization/`。如果需要单独运行旧方法，请使用独立入口：

```powershell
python legacy_main.py
```

可选旧流程：

```powershell
python legacy_main.py --list
python legacy_main.py --mode synthetic
python legacy_main.py --mode nsga2
python legacy_main.py --mode real-quick --scenario 5k
python legacy_main.py --mode real-stress
```

旧方法的输入、配置和输出都位于 `legacy_workflows/legacy_synthetic_optimization/` 内，不影响当前 `main.py` 主流程。旧流程主要用于追溯早期 CPU/GPU 异构资源调度实验，包括合成任务生成、FCFS/Price-Only/Homogeneous/GA/PSO/NSGA-II/Proposed 策略对比、消融实验、滚动实验、敏感性分析，以及真实缩放场景 quick/stress 测试。它用于对照和历史复查，不作为当前 NBSDC 三层融合与 Token 出口优化的主线入口。

## 模型说明

三层融合模型将集群级功率约束、服务器级任务负载、芯片级功率响应统一到小时尺度，构造以下指标：

- `cluster_cap_norm`：集群功率上限归一化值
- `server_load_norm`：服务器任务负载归一化值
- `chip_power_variation_norm`：芯片实际功率波动归一化值
- `chip_power_ratio`：芯片实际功率与芯片功率上限的比值，作为辅助指标

当前等效功率裕度公式为：

```text
power_margin_norm =
cluster_cap_norm
- 0.55 * server_load_norm
- 0.25 * chip_power_variation_norm
- 0.10
```

Token 产出能力由功率裕度折算：

```text
token_capacity_t = power_margin_norm_t * token_per_margin_unit
```

## Token 出口优化

RH-TEO 表示 Rolling-Horizon Token Export Optimization。项目构造国内、欧洲、北美三类 Token 服务场景，并综合收益、电力成本、跨时区时延、SLA 惩罚和出口波动惩罚进行滚动优化。

对比策略包括：

- `No-Export`
- `Power-Margin-Only`
- `Price-Driven`
- `Latency-Aware`
- `RH-TEO`

需要注意：NBSDC 原始数据不包含真实 Token 请求字段。本项目中的 Token 出口实验是基于三层融合得到的等效功率裕度构造的扩展场景，不应解释为真实线上 Token 业务记录。

## 项目结构

```text
config/                         Token 出口实验配置
data/raw/                       原始 Excel 输入，用户放置
data/real_case/                 清洗后的中间 CSV，本地生成
docs/                           实验设计和图表说明
experiments/                    核心实验逻辑
outputs/                        本地运行结果
scripts/                        数据清洗脚本
utils/                          IO、指标、绘图工具
main.py                         一键运行入口
run_nbsdc_fusion.py             分步运行入口
run_token_export.py             分步运行入口
legacy_workflows/legacy_synthetic_optimization/  旧版 synthetic 优化实验
```
