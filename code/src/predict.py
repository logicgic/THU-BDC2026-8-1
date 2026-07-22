"""Generate the top-five submission with the XGBoost Rank model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_FEATURE_TABLE, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT
from feature_engineering import build_feature_table
from xgb_rank_utils import load_feature_config, validate_feature_table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 XGBoost Rank 生成 Top5 提交")
    parser.add_argument("--input", default=DEFAULT_FEATURE_TABLE, help="feature_table.csv 或原始行情 CSV")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise RuntimeError("未安装 xgboost，请先运行 uv sync") from exc

    model_dir = Path(args.model_dir)
    feature_config_path = model_dir / "feature_config.json"
    if not feature_config_path.exists():
        raise FileNotFoundError(f"模型目录缺少冻结特征配置: {feature_config_path}")
    feature_config = load_feature_config(feature_config_path)
    feature_cols = feature_config["model_feature_cols"]

    raw = pd.read_csv(args.input, dtype={"股票代码": str, "instrument": str})
    if set(feature_cols).issubset(raw.columns) and {"datetime", "instrument"}.issubset(raw.columns):
        table = raw
    else:
        table, _ = build_feature_table(raw)
    table["datetime"] = pd.to_datetime(table["datetime"], errors="raise").dt.normalize()
    table["instrument"] = table["instrument"].astype("string").str.zfill(6)
    validate_feature_table(table, feature_cols, require_labels=False)
    latest_date = table["datetime"].max()
    latest = table.loc[table["datetime"] == latest_date].copy()
    if len(latest) < 5:
        raise ValueError(f"最新交易日可预测股票不足 5 只: {len(latest)}")

    model_path = model_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model = XGBRanker()
    model.load_model(str(model_path))
    trained_cols = model.get_booster().feature_names
    if trained_cols is not None and trained_cols != feature_cols:
        raise ValueError("模型特征列与 feature_config.json 不一致")
    latest["score"] = model.predict(latest[feature_cols])
    top5 = latest.sort_values(["score", "instrument"], ascending=[False, True]).head(5)
    stock_ids = top5["instrument"].astype(str).tolist()
    if len(stock_ids) != 5 or len(set(stock_ids)) != 5:
        raise AssertionError("预测结果必须包含 5 只不重复股票")
    output = pd.DataFrame({"stock_id": stock_ids, "weight": np.full(5, 0.2)})
    if not np.isfinite(output["weight"]).all() or not np.isclose(output["weight"].sum(), 1.0):
        raise AssertionError("输出权重无效")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"预测日期: {latest_date.date()}")
    print(f"参与排序股票数: {len(latest)}")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()
