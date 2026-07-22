"""XGBoost Rank training configuration."""

RANK_CONFIG = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 600,
    "early_stopping_rounds": 50,
    "random_state": 2026,
}

VALID_DAYS = 20
GAP_DAYS = 5
MIN_GROUP_SIZE = 10
TOP_K = 5
DEFAULT_FEATURE_TABLE = "data/features_xgb_rank/feature_table.csv"
DEFAULT_FEATURE_CONFIG = "data/features_xgb_rank/feature_config.json"
DEFAULT_MODEL_DIR = "model/xgb_rank"
DEFAULT_OUTPUT = "output/result.csv"
