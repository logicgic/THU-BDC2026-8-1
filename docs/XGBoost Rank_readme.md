# XGBoost Rank 项目流程

本文说明项目各模块、执行顺序和运行命令。当前项目已完成 XGBoost Rank 的特征工程基础模块；训练和推理入口仍需接入 XGBoost Rank 模型。

## 1. 环境准备

项目使用 Python 3.10-3.12，推荐使用项目虚拟环境：

```powershell
uv sync
.venv\Scripts\python.exe --version
```

## 2. 数据准备

### 2.1 获取行情数据

从数据源获取沪深 300 历史行情，并保存到 `data/stock_data.csv`：

```powershell
.venv\Scripts\python.exe get_stock_data.py `
  --start-date 2018-01-01 `
  --end-date 2026-12-31
```

该步骤会生成或更新股票池、行情缓存和失败清单。历史股票池应使用对应历史时点的成分股。

### 2.2 切分原始数据

按日期生成基础训练集和测试集：

```powershell
.venv\Scripts\python.exe data/split_train_test.py
```

默认输出：

- `data/train.csv`
- `data/test.csv`

## 3. 特征工程

特征工程模块位于 `code/src/feature_engineering.py`，处理顺序为：

1. 字段标准化、日期解析和股票代码补齐前导零。
2. 按股票和日期排序，检查重复行及异常负值。
3. 按股票计算官方技术因子。
4. 对每个交易日计算横截面 `rank_pct` 和 `zscore`。
5. 按股票向后取得 `open_t1`、`open_t5`。
6. 计算未来收益并按交易日分桶生成 `label_rank`。
7. 冻结模型特征列并保存配置。

运行命令：

```powershell
.venv\Scripts\python.exe code/src/build_feature_table.py `
  --input data/stock_data.csv `
  --output-dir data/features_xgb_rank `
  --min-group-size 10
```

输出：

- `data/features_xgb_rank/feature_table.csv`
- `data/features_xgb_rank/feature_config.json`

当前实际生成 28 个官方因子，模型特征共 84 列：原始值、横截面 rank、横截面 zscore 各 28 列。缺失值保留为 `NaN`，由 XGBoost 处理。

## 4. 时间切分

特征和标签生成完成后，才能划分训练集和验证集。禁止随机划分。

推荐配置：

- 训练集：较早日期。
- 隔离带：至少 5 个交易日。
- 验证集：最近 20 个交易日或最近 1 个月。
- 预测集：最新可用交易日，不参与训练和调参。

划分时检查训练集最大日期早于验证集最小日期，且两者之间存在完整隔离带。

单独生成切分产物：

```powershell
.venv\Scripts\python.exe code/src/split_rank_dataset.py
```

输出为 `model/xgb_rank/train_rank.csv`、`gap_rank.csv`、`valid_rank.csv` 和 `prediction_rank.csv`。训练入口也会执行相同切分并保存这些文件。

## 5. XGBoost Rank 训练

训练入口：

```powershell
.venv\Scripts\python.exe code/src/train.py `
  --feature-table data/features_xgb_rank/feature_table.csv `
  --feature-config data/features_xgb_rank/feature_config.json `
  --output-dir model/xgb_rank
```

训练要求：

- 使用 `XGBRanker(objective="rank:ndcg", eval_metric="ndcg@5")`。
- `qid` 使用交易日 `datetime`。
- 同一交易日的样本必须连续排列。
- 训练集和验证集按日期传入，不能随机抽样。
- 训练只使用 `model_feature_cols`，不得将标签字段放入特征。

预期产物：

- `model/xgb_rank/model.json`
- `model/xgb_rank/feature_config.json`
- `model/xgb_rank/train_config.json`
- `model/xgb_rank/metrics.json`

## 6. 验证

验证集至少记录以下指标：

- `ndcg@5`
- 每日预测 Top5 收益和
- 每日真实最优 Top5 收益和
- 预测收益与最优收益的比值
- Top5 平均收益和命中率

同时检查不同时间窗口的稳定性、换手率和最大回撤，不能只依据单次公开榜单结果调参。

## 7. 推理和提交

推理入口：

```powershell
.venv\Scripts\python.exe code/src/predict.py `
  --input data/stock_data.csv `
  --model-dir model/xgb_rank `
  --output output/result.csv
```

推理步骤：

1. 使用完整历史行情重新生成与训练一致的特征。
2. 读取冻结的 `model_feature_cols`，不得重新拟合处理器。
3. 取最新交易日全部候选股票并预测得分。
4. 按得分降序取前 5 只股票。
5. 输出 `stock_id,weight`，权重等权为 `0.2`。

提交前检查：

- 股票代码为字符串且不重复。
- 最多 5 只股票。
- 权重为有限数，总和在 `[0, 1]`，五只股票时总和接近 1。
- `output/result.csv` 路径、列名和编码正确。

## 8. 当前实现状态

| 模块 | 状态 | 入口 |
| --- | --- | --- |
| 行情获取 | 已有 | `get_stock_data.py` |
| 原始数据切分 | 已有 | `data/split_train_test.py` |
| XGBoost Rank 特征工程 | 已完成基础版本 | `code/src/build_feature_table.py` |
| 时间隔离切分 | 已接入训练流程 | `code/src/xgb_rank_utils.py` |
| XGBoost Rank 训练 | 已接入 | `code/src/train.py` |
| XGBoost Rank 验证 | 已接入 | `code/src/xgb_rank_utils.py` |
| XGBoost Rank 推理 | 已接入 | `code/src/predict.py` |
| 官方提交检查 | 已接入推理入口 | `test.sh` |

`train.sh` 会先生成完整特征表，再训练 XGBoost Rank；`test.sh` 使用训练产物生成 `output/result.csv`。旧版 Transformer 文件不再参与默认流程。
