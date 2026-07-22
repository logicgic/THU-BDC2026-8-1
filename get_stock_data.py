#!/usr/bin/env python3
"""Download point-in-time HS300 constituents and daily stock data.

BaoStock is the primary source.  AkShare is used only for individual quote
requests that still fail after BaoStock retries; it is never used to build a
historical constituent list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import baostock as bs
import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover - only reached with a broken environment
    ak = None


LOG = logging.getLogger("hs300-data")
OUTPUT_COLUMNS = [
    "股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
    "振幅", "涨跌额", "换手率", "涨跌幅",
]
PRICE_COLUMNS = ["开盘", "收盘", "最高", "最低"]
ZERO_FILL_COLUMNS = ["成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
NONNEGATIVE_COLUMNS = ["成交量", "成交额", "振幅", "换手率"]
QUOTE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
NO_ADJUSTMENT = "3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--output", default="data/stock_data.csv")
    parser.add_argument("--cache-dir", default="data/raw_cache")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--no-akshare-fallback", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def _normalise_date(value: str | date | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _pure_code(code: str) -> str:
    return str(code).lower().replace("sh.", "").replace("sz.", "").zfill(6)


def _bs_code(code: str) -> str:
    pure = _pure_code(code)
    return f"sh.{pure}" if pure.startswith(("5", "6", "9")) else f"sz.{pure}"


def _retry(
    operation: Callable[[], Any],
    name: str,
    retries: int,
    sleep_seconds: float,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as exc:  # APIs expose failures as both errors and exceptions.
            last_error = exc
            if attempt < retries:
                delay = sleep_seconds * attempt
                LOG.warning("%s failed (%s); retrying in %.1fs", name, exc, delay)
                time.sleep(delay)
    raise RuntimeError(f"{name} failed after {retries} attempts") from last_error


def _response_rows(response: Any, name: str) -> pd.DataFrame:
    if getattr(response, "error_code", "0") != "0":
        raise RuntimeError(f"{name} failed: {getattr(response, 'error_msg', '')}")
    rows: list[list[str]] = []
    while response.next():
        rows.append(response.get_row_data())
    return pd.DataFrame(rows, columns=response.fields)


def query_trade_dates(
    start_date: str,
    end_date: str,
    retries: int,
    sleep_seconds: float,
) -> list[str]:
    response = _retry(
        lambda: bs.query_trade_dates(start_date=start_date, end_date=end_date),
        "BaoStock trade calendar",
        retries,
        sleep_seconds,
    )
    frame = _response_rows(response, "BaoStock trade calendar")
    if frame.empty:
        raise RuntimeError("BaoStock returned an empty trade calendar")
    frame["is_trading_day"] = pd.to_numeric(frame["is_trading_day"], errors="coerce")
    return frame.loc[frame["is_trading_day"] == 1, "calendar_date"].map(_normalise_date).tolist()


def _snapshot_path(cache_dir: Path, snapshot_date: str) -> Path:
    return cache_dir / "membership" / f"{snapshot_date}.json"


def query_constituents(
    snapshot_date: str,
    cache_dir: Path,
    retries: int,
    sleep_seconds: float,
    force_refresh: bool = False,
) -> dict[str, str]:
    path = _snapshot_path(cache_dir, snapshot_date)
    if path.exists() and not force_refresh:
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

    response = _retry(
        lambda: bs.query_hs300_stocks(date=snapshot_date),
        f"BaoStock HS300 constituents {snapshot_date}",
        retries,
        sleep_seconds,
    )
    frame = _response_rows(response, f"BaoStock HS300 constituents {snapshot_date}")
    if frame.empty or not {"code", "code_name"}.issubset(frame.columns):
        raise RuntimeError(f"No valid HS300 constituents returned for {snapshot_date}")
    constituents = {_pure_code(row.code): str(row.code_name) for row in frame.itertuples()}
    if len(constituents) != 300:
        raise RuntimeError(f"Expected 300 constituents on {snapshot_date}, got {len(constituents)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(constituents, ensure_ascii=False, indent=2), encoding="utf-8")
    return constituents


def _first_trading_days(trading_days: list[str]) -> list[str]:
    return list(pd.Series(trading_days).groupby(pd.to_datetime(trading_days).to_period("M")).first())


def _find_transition(
    trading_days: list[str],
    left: int,
    right: int,
    left_set: set[str],
    right_set: set[str],
    query: Callable[[str], dict[str, str]],
) -> tuple[int, dict[str, str]]:
    """Find the first trading day carrying right_set between two probes."""
    while left + 1 < right:
        middle = (left + right) // 2
        middle_values = query(trading_days[middle])
        if set(middle_values) == left_set:
            left = middle
        else:
            right = middle
            right_set = set(middle_values)
    return right, query(trading_days[right])


def build_membership(
    trading_days: list[str],
    cache_dir: Path,
    retries: int,
    sleep_seconds: float,
    force_refresh: bool = False,
) -> pd.DataFrame:
    probes = _first_trading_days(trading_days)
    snapshots: dict[str, dict[str, str]] = {}
    query = lambda day: query_constituents(day, cache_dir, retries, sleep_seconds, force_refresh)
    for day in probes:
        snapshots[day] = query(day)

    probe_indices = [trading_days.index(day) for day in probes]
    for index in range(1, len(probes)):
        before = snapshots[probes[index - 1]]
        after = snapshots[probes[index]]
        if set(before) == set(after):
            continue
        transition_index, transition_values = _find_transition(
            trading_days,
            probe_indices[index - 1],
            probe_indices[index],
            set(before),
            set(after),
            query,
        )
        snapshots[trading_days[transition_index]] = transition_values

    ordered = sorted(snapshots.items(), key=lambda item: item[0])
    rows: list[dict[str, str]] = []
    for index, (effective, values) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else None
        end_date = (
            trading_days[trading_days.index(end) - 1]
            if end is not None
            else trading_days[-1]
        )
        for code, name in values.items():
            rows.append({
                "股票代码": code,
                "股票名称": name,
                "生效日期": effective,
                "失效日期": end_date,
            })
    membership = pd.DataFrame(rows).drop_duplicates()
    membership = membership.sort_values(["生效日期", "股票代码"]).reset_index(drop=True)
    _validate_membership(membership, trading_days)
    return membership


def _validate_membership(membership: pd.DataFrame, trading_days: list[str]) -> None:
    required = {"股票代码", "股票名称", "生效日期", "失效日期"}
    if membership.empty or not required.issubset(membership.columns):
        raise ValueError("Historical membership is empty or missing required columns")
    for code, group in membership.groupby("股票代码"):
        intervals = group.sort_values("生效日期")
        if intervals["生效日期"].duplicated().any():
            raise ValueError(f"Overlapping membership intervals for {code}")
        if (pd.to_datetime(intervals["生效日期"]) > pd.to_datetime(intervals["失效日期"])).any():
            raise ValueError(f"Invalid membership interval for {code}")
    for effective, group in membership.groupby("生效日期"):
        if len(group) != 300:
            LOG.warning("Membership on %s contains %d stocks, expected 300", effective, len(group))
    if membership["生效日期"].min() != trading_days[0]:
        raise ValueError("Membership does not cover the requested start date")


def _normalise_baostock(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["振幅"] = (frame["high"] - frame["low"]) / frame["preclose"] * 100
    frame["涨跌额"] = frame["close"] - frame["preclose"]
    frame["日期"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["股票代码"] = frame["code"].map(_pure_code)
    return frame.rename(columns={
        "open": "开盘", "close": "收盘", "high": "最高", "low": "最低",
        "volume": "成交量", "amount": "成交额", "turn": "换手率", "pctChg": "涨跌幅",
    })[OUTPUT_COLUMNS]


def fetch_baostock_history(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    response = bs.query_history_k_data_plus(
        _bs_code(code), QUOTE_FIELDS, start_date=start_date, end_date=end_date,
        frequency="d", adjustflag=NO_ADJUSTMENT,
    )
    frame = _response_rows(response, f"BaoStock quote {_pure_code(code)}")
    return _normalise_baostock(frame) if not frame.empty else pd.DataFrame(columns=OUTPUT_COLUMNS)


def fetch_akshare_history(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("AkShare is not installed")
    frame = ak.stock_zh_a_hist(
        symbol=_pure_code(code), period="daily", start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""), adjust="",
    )
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = frame.rename(columns={"日期": "日期", "开盘": "开盘", "收盘": "收盘", "最高": "最高", "最低": "最低"})
    frame["股票代码"] = _pure_code(code)
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame[OUTPUT_COLUMNS]


def clean_quote(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Apply the stable output contract to fresh and cached quote frames."""
    cleaned = frame.copy()
    required = set(OUTPUT_COLUMNS)
    if not required.issubset(cleaned.columns):
        missing = sorted(required - set(cleaned.columns))
        raise ValueError(f"Quote is missing columns: {missing}")
    cleaned["股票代码"] = cleaned["股票代码"].astype(str).str.zfill(6)
    if (~cleaned["股票代码"].str.fullmatch(r"\d{6}")).any():
        raise ValueError("Quote contains invalid stock codes")
    cleaned["日期"] = pd.to_datetime(cleaned["日期"], errors="coerce")
    if cleaned["日期"].isna().any():
        raise ValueError("Quote contains invalid dates")
    if start_date is not None and (cleaned["日期"] < pd.Timestamp(start_date)).any():
        raise ValueError("Quote contains dates before requested start date")
    if end_date is not None and (cleaned["日期"] > pd.Timestamp(end_date)).any():
        raise ValueError("Quote contains dates after requested end date")
    cleaned["日期"] = cleaned["日期"].dt.strftime("%Y-%m-%d")
    if cleaned.duplicated(["股票代码", "日期"]).any():
        raise ValueError("Quote contains duplicate stock/date rows")
    for column in PRICE_COLUMNS + ZERO_FILL_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    missing_prices = cleaned[PRICE_COLUMNS].isna().any(axis=1)
    if missing_prices.any():
        raise ValueError(f"Quote contains {int(missing_prices.sum())} rows with missing prices")
    if (~np.isfinite(cleaned[PRICE_COLUMNS].to_numpy())).any() or (cleaned[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Quote contains non-positive or non-finite prices")
    cleaned[ZERO_FILL_COLUMNS] = cleaned[ZERO_FILL_COLUMNS].fillna(0.0)
    if (~np.isfinite(cleaned[ZERO_FILL_COLUMNS].to_numpy())).any() or (cleaned[NONNEGATIVE_COLUMNS] < 0).any().any():
        raise ValueError("Quote contains negative or non-finite quantities")
    return cleaned[OUTPUT_COLUMNS]


def filter_by_membership(data: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    data = data.copy()
    data["日期_dt"] = pd.to_datetime(data["日期"])
    membership = membership.copy()
    membership["生效_dt"] = pd.to_datetime(membership["生效日期"])
    membership["失效_dt"] = pd.to_datetime(membership["失效日期"])
    filtered: list[pd.DataFrame] = []
    for code, stock_data in data.groupby("股票代码", sort=False):
        intervals = membership[membership["股票代码"] == code].sort_values("生效_dt")
        if intervals.empty:
            continue
        starts = intervals["生效_dt"].to_numpy()
        ends = intervals["失效_dt"].to_numpy()
        positions = starts.searchsorted(stock_data["日期_dt"].to_numpy(), side="right") - 1
        valid = positions >= 0
        valid[valid] = stock_data.loc[valid, "日期_dt"].to_numpy() <= ends[positions[valid]]
        filtered.append(stock_data.loc[valid, OUTPUT_COLUMNS])
    if not filtered:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(filtered, ignore_index=True).sort_values(
        ["股票代码", "日期"]
    ).reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_date = _normalise_date(args.start_date)
    end_date = _normalise_date(args.end_date)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")
    try:
        trading_days = query_trade_dates(start_date, end_date, args.max_retries, args.sleep_seconds)
        membership = build_membership(
            trading_days, cache_dir, args.max_retries, args.sleep_seconds, args.force_refresh
        )
        range_label = f"{start_date[:4]}_{end_date[:4]}"
        membership_path = output_path.parent / f"hs300_membership_{range_label}.csv"
        _atomic_csv(membership, membership_path)

        latest = membership[membership["失效日期"] == trading_days[-1]][["股票代码", "股票名称"]]
        latest = latest.rename(columns={"股票代码": "code", "股票名称": "code_name"})
        latest.insert(0, "updateDate", end_date)
        _atomic_csv(latest, output_path.parent / "hs300_stock_list.csv")

        quote_dir = cache_dir / "quotes"
        all_data: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        fallback_count = 0
        for index, code in enumerate(sorted(membership["股票代码"].unique()), start=1):
            cache_path = quote_dir / f"{code}.csv"
            try:
                if cache_path.exists() and not args.force_refresh:
                    quote = pd.read_csv(cache_path, dtype={"股票代码": str})
                else:
                    try:
                        quote = _retry(
                            lambda: fetch_baostock_history(code, start_date, end_date),
                            f"BaoStock quote {code}", args.max_retries, args.sleep_seconds,
                        )
                        if quote.empty:
                            raise RuntimeError("BaoStock returned no quote rows")
                    except Exception as primary_error:
                        if args.no_akshare_fallback:
                            raise
                        LOG.warning("Using AkShare fallback for %s: %s", code, primary_error)
                        quote = _retry(
                            lambda: fetch_akshare_history(code, start_date, end_date),
                            f"AkShare quote {code}", args.max_retries, args.sleep_seconds,
                        )
                        fallback_count += 1
                    _atomic_csv(quote, cache_path)
                quote = clean_quote(quote, start_date, end_date)
                all_data.append(quote)
                LOG.info("[%d] %s complete (%d rows)", index, code, len(quote))
            except Exception as exc:
                failures.append({"股票代码": code, "错误": str(exc)})
                LOG.error("%s failed: %s", code, exc)

        failed_path = output_path.parent / "failed_stocks.csv"
        _atomic_csv(pd.DataFrame(failures, columns=["股票代码", "错误"]), failed_path)
        if failures:
            raise RuntimeError(f"Failed to collect {len(failures)} stocks; see {failed_path}")
        combined = filter_by_membership(pd.concat(all_data, ignore_index=True), membership) if all_data else pd.DataFrame(columns=OUTPUT_COLUMNS)
        if combined.empty:
            raise RuntimeError("No quote data was collected")
        duplicates = combined.duplicated(["股票代码", "日期"]).sum()
        if duplicates:
            raise ValueError(f"Output contains {duplicates} duplicate stock/date rows")
        if output_path.exists():
            backup = output_path.with_name(
                f"{output_path.stem}.before_{range_label}_{datetime.now():%Y%m%d_%H%M%S}{output_path.suffix}"
            )
            shutil.copy2(output_path, backup)
            LOG.info("Backed up existing output to %s", backup)
        _atomic_csv(combined, output_path)
        LOG.info("Collected %d rows for %d stocks; AkShare fallback=%d; failures=%d", len(combined), combined["股票代码"].nunique(), fallback_count, len(failures))
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
