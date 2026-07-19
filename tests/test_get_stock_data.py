import pandas as pd

from get_stock_data import OUTPUT_COLUMNS, filter_by_membership


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
