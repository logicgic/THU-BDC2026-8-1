"""Shared data splitting, validation metrics, and artifact checks for XGBoost Rank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"open_t1", "open_t5", "label_return", "label_rank"}


def load_feature_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    cols = config.get("model_feature_cols")
    if not isinstance(cols, list) or not cols or len(cols) != len(set(cols)):
        raise ValueError("feature_config.json 的 model_feature_cols 无效或存在重复")
    if LABEL_COLUMNS.intersection(cols):
        raise ValueError("model_feature_cols 不得包含标签或未来价格列")
    return config


def validate_feature_table(df: pd.DataFrame, feature_cols: list[str], require_labels: bool = True) -> None:
    required = {"datetime", "instrument", *feature_cols}
    if require_labels:
        required.add("label_rank")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feature_table 缺少列: {sorted(missing)}")
    if df.duplicated(["datetime", "instrument"]).any():
        raise ValueError("feature_table 存在重复 datetime + instrument")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("特征列存在重复")
    values = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("特征中存在 inf")
    if require_labels:
        labels = pd.to_numeric(df["label_rank"].dropna(), errors="coerce")
        if labels.isna().any() or not labels.between(0, 4).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("label_rank 必须为 0~4 的整数")
        if "label_return" in df and np.isinf(pd.to_numeric(df["label_return"], errors="coerce")).any():
            raise ValueError("label_return 不得包含 inf")


def split_by_date(
    feature_table: pd.DataFrame,
    valid_days: int = 20,
    gap_days: int = 5,
    min_group_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split labeled rows into train/validation and keep latest prediction day."""
    if valid_days <= 0 or gap_days < 5 or min_group_size < 2:
        raise ValueError("valid_days 必须为正数，gap_days 至少为 5，min_group_size 至少为 2")
    feature_table = feature_table.copy()
    feature_table["datetime"] = pd.to_datetime(feature_table["datetime"], errors="raise").dt.normalize()
    labeled = feature_table.dropna(subset=["label_rank"]).copy()
    counts = labeled.groupby("datetime")["instrument"].transform("size")
    labeled = labeled.loc[counts >= min_group_size].copy()
    dates = np.array(sorted(labeled["datetime"].unique()), dtype="datetime64[ns]")
    if len(dates) <= valid_days + gap_days:
        raise ValueError("有效交易日不足以构造训练集、隔离带和验证集")
    valid_start_idx = len(dates) - valid_days
    train_end_idx = valid_start_idx - gap_days - 1
    if train_end_idx < 0:
        raise ValueError("训练集为空，请减少 valid_days 或 gap_days")
    train_end = pd.Timestamp(dates[train_end_idx])
    valid_start = pd.Timestamp(dates[valid_start_idx])
    train = labeled[labeled["datetime"] <= train_end].copy()
    gap_dates = dates[train_end_idx + 1:valid_start_idx]
    gap = feature_table[feature_table["datetime"].isin(gap_dates)].copy()
    valid = labeled[labeled["datetime"] >= valid_start].copy()
    prediction_date = pd.Timestamp(feature_table["datetime"].max())
    prediction = feature_table[feature_table["datetime"] == prediction_date].copy()
    train = train.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    gap = gap.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    valid = valid.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    prediction = prediction.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    train_dates = set(train["datetime"].unique())
    valid_dates = set(valid["datetime"].unique())
    if train_dates & valid_dates or train["datetime"].max() >= valid["datetime"].min():
        raise AssertionError("训练集和验证集日期重叠")
    if len(valid_dates) < valid_days:
        raise AssertionError("验证集有效交易日少于配置")
    if gap["datetime"].nunique() != gap_days:
        raise AssertionError("隔离带交易日数量与配置不一致")
    info = {
        "train_start": str(train["datetime"].min().date()),
        "train_end": str(train["datetime"].max().date()),
        "valid_start": str(valid["datetime"].min().date()),
        "valid_end": str(valid["datetime"].max().date()),
        "prediction_date": str(prediction_date.date()),
        "valid_days": valid_days,
        "gap_days": gap_days,
        "train_rows": len(train),
        "valid_rows": len(valid),
    }
    return train, gap, valid, prediction, info


def ranking_metrics(predictions: np.ndarray, df: pd.DataFrame, k: int = 5) -> dict[str, float]:
    values = np.asarray(predictions, dtype=float)
    if len(values) != len(df):
        raise ValueError("预测结果与验证数据行数不一致")
    work = df[["datetime", "label_rank", "label_return"]].copy()
    work["prediction"] = values
    rows: list[dict[str, float]] = []
    for _, day in work.groupby("datetime", sort=True):
        day = day.dropna(subset=["label_rank", "label_return"])
        if len(day) < k:
            continue
        n = min(k, len(day))
        pred = day.nlargest(n, "prediction")
        actual = day.nlargest(n, "label_return")
        relevance = day["label_rank"].to_numpy(dtype=float)
        order = day["prediction"].to_numpy(dtype=float).argsort()[::-1][:n]
        gains = relevance[order]
        discounts = np.log2(np.arange(2, n + 2))
        dcg = float(np.sum((2**gains - 1) / discounts))
        ideal = np.sort(relevance)[::-1][:n]
        idcg = float(np.sum((2**ideal - 1) / discounts))
        rows.append({
            "ndcg_at_5": dcg / idcg if idcg > 0 else 0.0,
            "pred_top5_return_sum": float(pred["label_return"].sum()),
            "best_top5_return_sum": float(actual["label_return"].sum()),
            "top5_mean_return": float(pred["label_return"].mean()),
            "return_ratio": float(pred["label_return"].sum() / (actual["label_return"].sum() + 1e-12)),
            "top5_hit_rate": float(len(set(pred.index) & set(actual.index)) / n),
        })
    if not rows:
        raise ValueError("没有足够的有效验证交易日")
    metrics = pd.DataFrame(rows).mean().to_dict()
    metrics["valid_days"] = float(len(rows))
    return {key: float(value) for key, value in metrics.items()}
