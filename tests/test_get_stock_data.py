from unittest.mock import Mock, patch

import pandas as pd
import pytest

from get_stock_data import (
    OUTPUT_COLUMNS,
    clean_quote,
    fetch_akshare_history,
    fetch_baostock_history,
    filter_by_membership,
)


def test_filter_by_membership_respects_effective_boundaries() -> None:
    quotes = pd.DataFrame(
        [
            {"股票代码": "000001", "日期": "2018-01-02", **{column: 1 for column in OUTPUT_COLUMNS[2:]}},
            {"股票代码": "000001", "日期": "2018-02-01", **{column: 2 for column in OUTPUT_COLUMNS[2:]}},
            {"股票代码": "000002", "日期": "2018-01-02", **{column: 3 for column in OUTPUT_COLUMNS[2:]}},
        ]
    )
    membership = pd.DataFrame(
        [
            {"股票代码": "000001", "股票名称": "A", "生效日期": "2018-01-01", "失效日期": "2018-01-31"},
            {"股票代码": "000002", "股票名称": "B", "生效日期": "2018-02-01", "失效日期": "2018-12-31"},
        ]
    )

    result = filter_by_membership(quotes, membership)

    assert result["日期"].tolist() == ["2018-01-02"]
    assert result["股票代码"].tolist() == ["000001"]


def test_filter_returns_required_columns_and_sorted_rows() -> None:
    row = {"股票代码": "000002", "日期": "2018-01-03"}
    row.update({column: 1 for column in OUTPUT_COLUMNS[2:]})
    quotes = pd.DataFrame([row])
    membership = pd.DataFrame(
        [{"股票代码": "000002", "股票名称": "B", "生效日期": "2018-01-01", "失效日期": "2018-12-31"}]
    )

    result = filter_by_membership(quotes, membership)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.iloc[0]["股票代码"] == "000002"


def _quote_row(**overrides: object) -> dict[str, object]:
    row = {"股票代码": "000001", "日期": "2018-01-02"}
    row.update({column: 1 for column in OUTPUT_COLUMNS[2:]})
    row.update(overrides)
    return row


def test_clean_quote_rejects_invalid_rows() -> None:
    with pytest.raises(ValueError, match="invalid stock codes"):
        clean_quote(pd.DataFrame([_quote_row(股票代码="bad")]))
    with pytest.raises(ValueError, match="invalid dates"):
        clean_quote(pd.DataFrame([_quote_row(日期="bad")]))
    with pytest.raises(ValueError, match="non-positive"):
        clean_quote(pd.DataFrame([_quote_row(开盘=0)]))
    with pytest.raises(ValueError, match="duplicate"):
        clean_quote(pd.DataFrame([_quote_row(), _quote_row()]))


def test_fetch_sources_use_unadjusted_data() -> None:
    bs_response = Mock(error_code="0", fields=OUTPUT_COLUMNS)
    bs_response.next.side_effect = [False]
    with patch("get_stock_data.bs.query_history_k_data_plus", return_value=bs_response) as query:
        fetch_baostock_history("000001", "2018-01-01", "2018-01-02")
    assert query.call_args.kwargs["adjustflag"] == "3"

    ak_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    with patch("get_stock_data.ak.stock_zh_a_hist", return_value=ak_frame) as query:
        fetch_akshare_history("000001", "2018-01-01", "2018-01-02")
    assert query.call_args.kwargs["adjust"] == ""
