"""Data quality checks and issue extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_COLUMNS = ["Grid Value", "DG Value", "Battery Value", "Total Value"]


def reconciliation_delta(df: pd.DataFrame) -> pd.Series:
    parts = df["Grid Value"].fillna(0) + df["DG Value"].fillna(0) + df["Battery Value"].fillna(0)
    total = df["Total Value"]
    return (parts - total).abs()


def run_quality_report(
    df: pd.DataFrame,
    *,
    pct_min: float = 0.0,
    pct_max: float = 100.0,
    reconcile_atol: float = 1.25,
    reconcile_rtol: float = 0.01,
) -> dict:
    """
    Return summary counts and flags. Reconciliation uses atol/rtol similar to numpy.isclose
    (scaled for percentage rounding in source data).
    """
    key = ["month_label", "Site ID"]
    dup = df.duplicated(subset=key, keep=False)

    nulls = {c: float(df[c].isna().mean()) for c in df.columns if c in VALUE_COLUMNS or c == "month_label"}

    range_issues = pd.Series(False, index=df.index)
    for c in VALUE_COLUMNS:
        v = df[c]
        bad = v.notna() & ((v < pct_min) | (v > pct_max))
        range_issues = range_issues | bad

    delta = reconciliation_delta(df)
    total = df["Total Value"].replace(0, np.nan)
    tol = reconcile_atol + reconcile_rtol * total.abs()
    tol = tol.fillna(reconcile_atol)
    reconcile_fail = delta.notna() & total.notna() & (delta > tol)

    return {
        "n_rows": int(len(df)),
        "n_duplicates_key": int(dup.sum()),
        "null_rate_by_col": nulls,
        "n_range_violations": int(range_issues.sum()),
        "n_reconcile_failures": int(reconcile_fail.sum()),
        "reconcile_failure_rate": float(reconcile_fail.mean()) if len(df) else 0.0,
        "duplicate_mask": dup,
        "range_mask": range_issues,
        "reconcile_mask": reconcile_fail,
        "reconcile_delta": delta,
    }


def reconciliation_by_month(df: pd.DataFrame, rec_mask: pd.Series) -> pd.DataFrame:
    t = df.assign(_fail=rec_mask.astype(int))
    g = t.groupby("month_label", sort=True)._fail.agg(["sum", "count"]).reset_index()
    g["fail_rate"] = g["sum"] / g["count"].replace(0, np.nan)
    return g
