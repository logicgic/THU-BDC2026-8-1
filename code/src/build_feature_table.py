"""Build and persist the XGBoost Rank feature table."""

from __future__ import annotations

import argparse

import pandas as pd

from feature_engineering import build_feature_table, save_feature_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/stock_data.csv")
    parser.add_argument("--output-dir", default="data/features_xgb_rank")
    parser.add_argument("--min-group-size", type=int, default=10)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, dtype={"股票代码": str, "instrument": str})
    table, model_cols = build_feature_table(raw, min_group_size=args.min_group_size)
    save_feature_artifacts(table, model_cols, args.output_dir)
    labeled = table.dropna(subset=["label_rank"])
    print(f"feature_table rows={len(table)}, model_features={len(model_cols)}, labeled_rows={len(labeled)}")


if __name__ == "__main__":
    main()

