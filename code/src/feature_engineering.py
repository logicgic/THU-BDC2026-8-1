"""Leakage-aware feature engineering for the XGBoost learning-to-rank model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import engineer_features_39


RAW_FEATURE_COLS = [
    "sma_5", "sma_20", "ema_12", "ema_26", "rsi", "macd", "macd_signal",
    "volume_change", "obv", "volume_ma_5", "volume_ma_20", "volume_ratio",
    "kdj_k", "kdj_d", "kdj_j", "boll_mid", "boll_std", "atr_14", "ema_60",
    "volatility_10", "volatility_20", "return_1", "return_5", "return_10",
    "high_low_spread", "open_close_spread", "high_close_spread", "low_close_spread",
]
LABEL_COLUMNS = ["open_t1", "open_t5", "label_return", "label_rank"]
REQUIRED_COLUMNS = ["datetime", "instrument", "open", "close", "high", "low", "volume", "amount"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "日期": "datetime", "股票代码": "instrument", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    out = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns}).copy()
    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"行情数据缺少必要列: {sorted(missing)}")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    out["instrument"] = out["instrument"].astype("string").str.strip().str.zfill(6)
    numeric = ["open", "close", "high", "low", "volume", "amount"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    out = out.dropna(subset=["datetime", "instrument"])
    out = out.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    if out.duplicated(["instrument", "datetime"]).any():
        raise ValueError("存在重复的 instrument + datetime 行")
    if (out[["open", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("开收高低价必须为正数")
    if (out["volume"] < 0).any():
        raise ValueError("成交量不得为负数")
    return out


def _engineer_by_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute factors on the full per-stock history before any time split."""
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("instrument", sort=False, observed=True):
        work = group.rename(columns={"open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额"})
        work = engineer_features_39(work)
        parts.append(work)
    result = pd.concat(parts, ignore_index=True)
    result = result.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"})
    result = result.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    result[RAW_FEATURE_COLS] = result[RAW_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return result


def _add_cross_sectional_transforms(df: pd.DataFrame, feature_cols: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    rank_cols: list[str] = []
    zscore_cols: list[str] = []
    grouped = out.groupby("datetime", observed=True)
    for col in feature_cols:
        rank_col = f"{col}_rank_pct"
        zscore_col = f"{col}_zscore"
        out[rank_col] = grouped[col].rank(pct=True)
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std")
        out[zscore_col] = ((out[col] - mean) / (std + 1e-12)).where(std > 0, 0.0)
        rank_cols.append(rank_col)
        zscore_cols.append(zscore_col)
    return out, list(feature_cols) + rank_cols + zscore_cols


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["instrument", "datetime"]).copy()
    grouped = out.groupby("instrument", sort=False, observed=True)["open"]
    out["open_t1"] = grouped.shift(-1)
    out["open_t5"] = grouped.shift(-5)
    out["label_return"] = (out["open_t5"] - out["open_t1"]) / out["open_t1"]
    out["label_return"] = out["label_return"].replace([np.inf, -np.inf], np.nan)
    pct = out.groupby("datetime", observed=True)["label_return"].rank(pct=True)
    out["label_rank"] = 2
    out.loc[pct >= 0.95, "label_rank"] = 4
    out.loc[(pct >= 0.80) & (pct < 0.95), "label_rank"] = 3
    out.loc[(pct > 0.05) & (pct <= 0.20), "label_rank"] = 1
    out.loc[pct <= 0.05, "label_rank"] = 0
    out.loc[out["label_return"].isna(), "label_rank"] = pd.NA
    out["label_rank"] = out["label_rank"].astype("Int64")
    return out.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def build_feature_table(df: pd.DataFrame, min_group_size: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """Return a complete feature table and frozen model feature columns."""
    normalized = _standardize_columns(df)
    result = _engineer_by_stock(normalized)
    result, model_feature_cols = _add_cross_sectional_transforms(result, RAW_FEATURE_COLS)
    result = _add_labels(result)
    counts = result.dropna(subset=["label_rank"]).groupby("datetime")["instrument"].transform("size")
    result.loc[result["label_rank"].notna() & (counts < min_group_size), "label_rank"] = pd.NA
    result[model_feature_cols] = result[model_feature_cols].apply(pd.to_numeric, errors="coerce")
    if len(model_feature_cols) != len(set(model_feature_cols)):
        raise AssertionError("model_feature_cols 存在重复列")
    if np.isinf(result[model_feature_cols].to_numpy(dtype=float, na_value=np.nan)).any():
        raise AssertionError("model_feature_cols 仍包含 inf")
    return result, model_feature_cols


def save_feature_artifacts(feature_table: pd.DataFrame, model_feature_cols: list[str], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(output / "feature_table.csv", index=False)
    config = {
        "feature_version": "xgboost_rank_v1",
        "raw_feature_cols": RAW_FEATURE_COLS,
        "model_feature_cols": model_feature_cols,
        "label_definition": "(open_t5-open_t1)/open_t1; daily percentile buckets 0..4",
        "date_min": str(feature_table["datetime"].min().date()),
        "date_max": str(feature_table["datetime"].max().date()),
    }
    (output / "feature_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
