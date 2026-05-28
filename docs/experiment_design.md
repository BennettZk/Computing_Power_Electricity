# 实验设计说明

## 一键复现实验链路

当前项目以 `main.py` 作为唯一推荐入口。只要在 `data/raw/` 放入三份原始 Excel 文件，就可以在项目根目录运行：

```powershell
python main.py
```

完整链路为：

1. 自动识别 `data/raw/` 中的集群级、服务器级、芯片级数据。
2. 清洗原始 Excel，生成 `data/real_case/` 下的标准 CSV。
3. 将三层数据对齐到小时尺度，生成 NBSDC 融合指标。
4. 将等效功率裕度折算为 Token 产出能力。
5. 运行 Token 出口策略对比、RH-TEO 优化和参数敏感性实验。

`scripts/prepare_uploaded_case_dataset.py`、`run_nbsdc_fusion.py` 和 `run_token_export.py` 仍可用于分步调试，但论文复现实验应优先描述 `main.py`。

旧版 synthetic 优化方法保留在 `legacy_workflows/legacy_synthetic_optimization/`，并通过根目录 `legacy_main.py` 单独运行。该入口只用于复查旧方法，不作为当前主线实验入口：

```powershell
python legacy_main.py --list
python legacy_main.py --mode synthetic
python legacy_main.py --mode real-quick --scenario 5k
```

## 三类数据的综合利用逻辑

项目将 NBSDC 三类数据按物理层级串联：

- 集群级数据提供电价、功率上限和 Power Capping 场景，表示数据中心可用电力约束。
- 服务器级数据提供任务到达、CPU 需求、deadline 和机房/机架/服务器层级分布，表示业务负载压力。
- 芯片级数据提供 DVFS 频率、实际功率、报量功率和芯片功率上限，表示设备侧功率响应。

三层数据统一到小时尺度后，构造 `cluster_cap_norm`、`server_load_norm`、`chip_power_variation_norm` 和 `power_margin_norm`。其中 `power_margin_norm` 是融合后的等效指标，不是 NBSDC 原始字段。

## 章节对应关系

建议论文结构如下：

| 章节 | 内容 |
| --- | --- |
| 第 3 章 | NBSDC 多层级数据清洗与三层融合建模 |
| 第 4 章 | 基于等效功率裕度的 Token 出口优化模型 |
| 第 5 章 | 策略对比、敏感性分析和结果讨论 |

第 3 章重点说明数据来源、字段处理、小时级对齐和功率裕度构造。第 4 章重点说明 Token 产能折算、区域服务场景、目标函数和 RH-TEO 滚动优化。第 5 章重点展示实验结果和模型边界。

## 与任务书要求的对应关系

- 使用真实数据：清洗并融合 NBSDC 三个原始 Excel 文件。
- 不只做单一算法：对比 `No-Export`、`Power-Margin-Only`、`Price-Driven`、`Latency-Aware` 和 `RH-TEO`。
- 具备工程链路：从原始 Excel 到标准 CSV、融合指标、优化结果和图表输出均可一键运行。
- 具备可解释性：功率裕度、Token 产能和区域出口策略之间的关系在代码、README 和图表中保持一致。

## Token 出口场景设定

项目以中国数据中心为算力供给侧，构造国内、欧洲、北美三类 Token 服务场景：

| 场景 | 定位 | 时延特征 | 价格特征 |
| --- | --- | --- | --- |
| 国内 | 本地或近域需求 | 低时延 | 低价格 |
| 欧洲 | 中距离跨境出口 | 中等时延 | 中等价格 |
| 北美 | 远端高价格出口市场 | 高时延 | 高价格 |

欧洲和北美参数是扩展场景假设，不是 NBSDC 原始字段。价格敏感性实验选取北美作为高价格、长时延远端市场代表，用于分析远端市场价格变化对 Token 出口分配和净收益的影响。

## 模型假设与边界

- NBSDC 不包含真实 Token 请求记录。
- Token 出口是基于三层融合结果构造的扩展场景。
- 等效功率裕度是归一化建模指标，不等同于真实剩余功率。
- 国内、欧洲、北美参数用于刻画价格和时延差异，不是 NBSDC 原始字段。
- 不应将原始任务强行解释为 CPU/GPU/token batch/remote pool 请求。

## 输出管理

`data/real_case/`、`outputs/`、`legacy_workflows/legacy_synthetic_optimization/data/` 和 `legacy_workflows/legacy_synthetic_optimization/outputs/` 都是运行产物目录，不参与版本管理。仓库通过 `.gitkeep` 保留目录结构，通过 `.gitignore` 忽略实际数据和结果文件。

这意味着论文复现时的正确流程是：准备 `data/raw/` 原始 Excel，运行 `main.py`，再从 `outputs/` 中取图表和结果表。
