"""Train the project's XGBoost learning-to-rank model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_TABLE,
    DEFAULT_MODEL_DIR,
    GAP_DAYS,
    MIN_GROUP_SIZE,
    RANK_CONFIG,
    VALID_DAYS,
)
from xgb_rank_utils import load_feature_config, ranking_metrics, split_by_date, validate_feature_table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 XGBoost Rank 排序模型")
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--valid-days", type=int, default=VALID_DAYS)
    parser.add_argument("--gap-days", type=int, default=GAP_DAYS)
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    return parser.parse_args()


def _qid(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.factorize(df["datetime"], sort=True)[0], index=df.index, dtype="int64")


def _fit_ranker(model, train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str]) -> None:
    fit_args = {
        "eval_set": [(valid[feature_cols], valid["label_rank"].astype(int))],
        "verbose": False,
    }
    try:
        model.fit(
            train[feature_cols],
            train["label_rank"].astype(int),
            qid=_qid(train),
            eval_qid=[_qid(valid)],
            **fit_args,
        )
    except TypeError as exc:
        if "qid" not in str(exc):
            raise
        group_train = train.groupby("datetime", sort=False).size().to_numpy()
        group_valid = valid.groupby("datetime", sort=False).size().to_numpy()
        model.fit(
            train[feature_cols],
            train["label_rank"].astype(int),
            group=group_train,
            eval_group=[group_valid],
            **fit_args,
        )


def main() -> None:
    args = _parse_args()
    try:
        import xgboost
        from xgboost import XGBRanker
    except ImportError as exc:
        raise RuntimeError("未安装 xgboost，请先运行 uv sync") from exc

    feature_config = load_feature_config(args.feature_config)
    feature_cols = feature_config["model_feature_cols"]
    table = pd.read_csv(args.feature_table, dtype={"instrument": str})
    table["datetime"] = pd.to_datetime(table["datetime"], errors="raise").dt.normalize()
    table["instrument"] = table["instrument"].astype("string").str.zfill(6)
    validate_feature_table(table, feature_cols)
    train, gap, valid, prediction, split_info = split_by_date(
        table,
        valid_days=args.valid_days,
        gap_days=args.gap_days,
        min_group_size=args.min_group_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_rank.csv", index=False)
    gap.to_csv(output_dir / "gap_rank.csv", index=False)
    valid.to_csv(output_dir / "valid_rank.csv", index=False)
    prediction.to_csv(output_dir / "prediction_rank.csv", index=False)
    frozen_config_path = output_dir / "feature_config.json"
    if Path(args.feature_config).resolve() != frozen_config_path.resolve():
        shutil.copyfile(args.feature_config, frozen_config_path)

    params = dict(RANK_CONFIG)
    model = XGBRanker(**params)
    _fit_ranker(model, train, valid, feature_cols)
    model.save_model(str(output_dir / "model.json"))

    metrics = ranking_metrics(model.predict(valid[feature_cols]), valid)
    metrics.update(split_info)
    metrics["best_iteration"] = int(getattr(model, "best_iteration", params["n_estimators"] - 1))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    train_config = {
        "model_params": params,
        "split": split_info,
        "feature_count": len(feature_cols),
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (output_dir / "train_config.json").write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
