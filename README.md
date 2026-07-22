# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：完整历史行情生成的横截面技术特征；
- 模型：`XGBRanker`，按交易日学习股票横截面排序；
- 输出：对同一天全部候选股票打分并排序，最终输出前5只股票（等权重0.2）。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取完整历史行情数据（`data/stock_data.csv`）；
2. 生成官方技术因子及同日 rank/zscore 特征；
3. 构建未来开盘收益和按日分桶的 `label_rank`；
4. 按交易日切分训练集、5日隔离带、验证集和预测日；
5. 使用 `XGBRanker` 训练并计算 `ndcg@5` 与 Top5 收益指标；
6. 对最新交易日预测并生成 `output/result.csv`。

---

## 2. 代码结构说明

### [config.py](config.py)
统一管理训练与推理参数，包括：
- XGBoost Rank 树模型参数；
- 验证集 20 个交易日及 5 个交易日隔离带；
- 最小 query 股票数、模型目录和提交路径。

### [feature_engineering.py](code/src/feature_engineering.py)
生成 28 个官方技术因子，以及同日横截面 rank/zscore 转换，构造 `label_rank` 并冻结模型特征列。

### [utils.py](utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [train.py](code/src/train.py)
使用交易日作为 `qid` 训练 `XGBRanker`，保存模型、切分数据、训练配置和验证指标。

### [predict.py](code/src/predict.py)
读取冻结特征列和 XGBoost 模型，对最新交易日排序，输出 `stock_id,weight` 到 `output/result.csv`。

### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（BaoStock 主源，AkShare 单股备用）：
- 按交易日查询历史沪深300成分股，避免用当前成分股回填历史；
- 使用 BaoStock 后复权日线，单只股票请求失败时切换 AkShare；
- 每只股票独立缓存，支持重试、断点续抓和失败清单；
- 输出 `data/hs300_membership_2018_2026.csv`、`data/stock_data.csv` 和 `data/failed_stocks.csv`。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- output目录下 `result.csv`（由 `predict.py` 生成）。

---

## 4. 运行方法（推荐使用 uv）

1) 使用 `uv` 安装依赖

`uv sync`

抓取 2018—2026 年数据：

```powershell
uv run python get_stock_data.py --start-date 2018-01-01 --end-date 2026-12-31
```

2) 激活虚拟环境

`source .venv/bin/activate`

3) 训练模型

```
sh train.sh
```

4) 生成预测结果

```
sh test.sh
```

---

## 5. 常见问题

1) `TA-Lib` 安装失败  
本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装Python包。
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

2) 多进程相关问题  
`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS下请保持通过脚本入口运行（不要在交互式环境里直接多进程调用主逻辑）。

3) GPU/CPU自动选择  
代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无GPU时可直接CPU运行。
