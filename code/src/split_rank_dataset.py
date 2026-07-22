"""Persist the chronological train/gap/validation/prediction datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_FEATURE_CONFIG, DEFAULT_FEATURE_TABLE, DEFAULT_MODEL_DIR, GAP_DAYS, MIN_GROUP_SIZE, VALID_DAYS
from xgb_rank_utils import load_feature_config, split_by_date, validate_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="按交易日切分 XGBoost Rank 数据集")
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--valid-days", type=int, default=VALID_DAYS)
    parser.add_argument("--gap-days", type=int, default=GAP_DAYS)
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    args = parser.parse_args()
    config = load_feature_config(args.feature_config)
    table = pd.read_csv(args.feature_table, dtype={"instrument": str})
    table["datetime"] = pd.to_datetime(table["datetime"], errors="raise").dt.normalize()
    validate_feature_table(table, config["model_feature_cols"])
    train, gap, valid, prediction, info = split_by_date(
        table, args.valid_days, args.gap_days, args.min_group_size
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("train_rank", train), ("gap_rank", gap), ("valid_rank", valid), ("prediction_rank", prediction)):
        data.to_csv(output / f"{name}.csv", index=False)
    print(info)


if __name__ == "__main__":
    main()

