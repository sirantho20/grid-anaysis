"""Pure aggregation helpers for filtering and rollups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

THREE_SOURCES = ["Grid Value", "DG Value", "Battery Value"]

_SOURCE_LABELS = {"Grid Value": "Grid", "DG Value": "DG (genset)", "Battery Value": "Battery"}

RUNTIME_SOURCES = ["Grid_runtime", "DG_runtime", "Battery_runtime"]
TOTAL_RUNTIME_COL = "Total_runtime"


def _runtime_cap_per_row(total: pd.Series) -> pd.Series:
    """Hours cap per site-month row: ``min(positive Total_runtime, 24)`` else 24."""
    t = pd.to_numeric(total, errors="coerce")
    cap = pd.Series(24.0, index=t.index, dtype=float)
    ok = t.notna() & (t > 0)
    cap.loc[ok] = t.loc[ok].clip(upper=24.0)
    return cap


def _runtime_frame_for_averaging(d: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize runtime columns so Grid + DG + Battery does not exceed the per-row cap,
    scaling sources proportionally when needed; ``Total_runtime`` clipped to cap.

    Source values are interpreted as daily hours per site-month; aggregations take **means**.
    """
    if d.empty:
        return d
    cols_src = RUNTIME_SOURCES
    needed = cols_src + [TOTAL_RUNTIME_COL]
    if any(c not in d.columns for c in needed):
        return d.copy()

    out = d.copy()
    src_df = out[cols_src].apply(pd.to_numeric, errors="coerce")
    source_sum = src_df.fillna(0).sum(axis=1)
    cap = _runtime_cap_per_row(out[TOTAL_RUNTIME_COL])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (source_sum.to_numpy(dtype=float) > cap.to_numpy(dtype=float))
            & (source_sum.to_numpy(dtype=float) > 0),
            (cap / source_sum).to_numpy(dtype=float),
            1.0,
        )
    for c in cols_src:
        out[c] = np.where(src_df[c].notna(), src_df[c].to_numpy(dtype=float) * ratio, np.nan)

    tot = pd.to_numeric(out[TOTAL_RUNTIME_COL], errors="coerce")
    out[TOTAL_RUNTIME_COL] = np.minimum(tot.to_numpy(dtype=float), cap.to_numpy(dtype=float))
    return out


def month_list(df: pd.DataFrame) -> list[str]:
    return sorted(df["month_label"].dropna().unique().tolist())


def row_avg3(df: pd.DataFrame) -> pd.Series:
    """Mean of Grid, DG, Battery per row (availability %)."""
    return df[THREE_SOURCES].mean(axis=1)


def apply_filters(
    df: pd.DataFrame,
    months: list[str] | None,
    du_coop: list[str] | None,
    province: list[str] | None,
    power_model: list[str] | None,
    portfolio: list[str] | None,
    site_search: str | None,
) -> pd.DataFrame:
    d = df
    if months:
        d = d[d["month_label"].isin(months)]
    if du_coop:
        d = d[d["DU COOP"].isin(du_coop)]
    if province:
        d = d[d["Province"].isin(province)]
    if power_model:
        d = d[d["Power_Model"].isin(power_model)]
    if portfolio:
        d = d[d["Portfolio"].isin(portfolio)]
    if site_search and str(site_search).strip():
        q = str(site_search).strip().lower()
        mask = d["Site ID"].astype(str).str.lower().str.contains(q, na=False) | d[
            "Site Name"
        ].astype(str).str.lower().str.contains(q, na=False)
        d = d[mask]
    return d


def portfolio_means_by_month(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("month_label", sort=True)[THREE_SOURCES].mean().reset_index()
    g["month_ts"] = pd.to_datetime(g["month_label"] + "-01")
    return g.sort_values("month_ts")


def mix_shares_by_month(d: pd.DataFrame) -> pd.DataFrame:
    """Per site-month, share of each source vs (Grid+DG+Battery); then mean share by month."""
    x = d[d[THREE_SOURCES].sum(axis=1) > 0].copy()
    denom = x[THREE_SOURCES].sum(axis=1)
    for c in THREE_SOURCES:
        x[c + "_share"] = x[c] / denom
    cols = [c + "_share" for c in THREE_SOURCES]
    g = x.groupby("month_label", sort=True)[cols].mean().reset_index()
    g["month_ts"] = pd.to_datetime(g["month_label"] + "-01")
    return g.sort_values("month_ts")


def mom_changes(means: pd.DataFrame) -> pd.DataFrame:
    m = means.sort_values("month_ts").copy()
    for c in THREE_SOURCES:
        if c in m.columns:
            m[c + "_mom"] = m[c].diff()
    return m


def correlation_matrix(d: pd.DataFrame) -> pd.DataFrame:
    return d[THREE_SOURCES].corr(numeric_only=True)


def site_diagnostics(
    d: pd.DataFrame, *, top_n: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Laggards on avg3, per-site stats on avg3, heatmaps for each source (top-N volatile by std avg3)."""
    if d.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    latest = d["month_label"].max()
    latest_df = d[d["month_label"] == latest].copy()
    latest_df["avg3"] = row_avg3(latest_df)

    show_cols = ["Site ID", "Site Name", "Province", "DU COOP", "Grid Value", "DG Value", "Battery Value", "avg3"]
    latest_laggards = latest_df.nsmallest(min(15, len(latest_df)), "avg3")[show_cols]

    rows: list[dict[str, Any]] = []
    for sid, g in d.groupby("Site ID", sort=False):
        g = g.sort_values("month_ts")
        a = row_avg3(g)
        row_latest = g[g["month_label"] == latest]
        latest_avg3 = float(row_avg3(row_latest).mean()) if len(row_latest) else float("nan")
        roll3_avg3 = float(a.tail(3).mean())
        mean_avg3 = float(a.mean())
        std_avg3 = float(a.std(ddof=0)) if len(g) > 1 else 0.0
        diffs = a.diff()
        max_drop = float((-diffs).max()) if len(g) > 1 else 0.0
        rows.append(
            {
                "Site ID": sid,
                "Site Name": g["Site Name"].iloc[-1],
                "DU COOP": g["DU COOP"].iloc[-1],
                "Province": g["Province"].iloc[-1],
                "latest_avg3": latest_avg3,
                "roll3_avg3": roll3_avg3,
                "mean_avg3": mean_avg3,
                "std_avg3": std_avg3,
                "cv": std_avg3 / mean_avg3 if mean_avg3 else np.nan,
                "max_month_drop": max_drop,
                "n_months": int(g["month_label"].nunique()),
            }
        )

    vol = pd.DataFrame(rows)
    heat_ids = vol.nlargest(min(top_n, len(vol)), "std_avg3")["Site ID"].astype(str).tolist()
    sub = d[d["Site ID"].astype(str).isin(heat_ids)]
    heats = []
    for col in THREE_SOURCES:
        h = sub.pivot_table(index="Site ID", columns="month_label", values=col, aggfunc="mean")
        col_order = sorted(h.columns)
        heats.append(h[col_order] if len(h.columns) else h)
    return latest_laggards, vol, heats[0], heats[1], heats[2]


def geo_mean_bar_long(
    d: pd.DataFrame,
    dim: str,
    *,
    month_focus: str | None = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Long-form means of Grid/DG/Battery for grouped bar chart (top categories by mean avg3)."""
    x = d.copy()
    if month_focus:
        x = x[x["month_label"] == month_focus]
    g = x.groupby(dim, dropna=False)[THREE_SOURCES].mean().reset_index()
    g["_avg3"] = g[THREE_SOURCES].mean(axis=1)
    g = g.nlargest(top_n, "_avg3").drop(columns=["_avg3"])
    long = g.melt(id_vars=[dim], value_vars=THREE_SOURCES, var_name="source", value_name="availability")
    long["source"] = long["source"].map(_SOURCE_LABELS).fillna(long["source"])
    return long


def du_coop_trend_long(d: pd.DataFrame, *, top_k: int = 8) -> pd.DataFrame:
    """Long-form monthly means by DU COOP for faceted line chart (one row per source)."""
    if d.empty:
        return pd.DataFrame(columns=["month_ts", "DU COOP", "source", "availability"])
    coop_order = d.groupby("DU COOP")["Site ID"].nunique().sort_values(ascending=False).head(top_k).index.tolist()
    sub = d[d["DU COOP"].isin(coop_order)]
    parts = []
    for col in THREE_SOURCES:
        g = sub.groupby(["month_ts", "DU COOP"], sort=True)[col].mean().reset_index()
        g["source"] = _SOURCE_LABELS[col]
        g = g.rename(columns={col: "availability"})
        parts.append(g[["month_ts", "DU COOP", "source", "availability"]])
    return pd.concat(parts, ignore_index=True)


def du_coop_grid_rankings(
    df: pd.DataFrame,
    *,
    worst_n: int = 5,
) -> dict[str, Any]:
    """
    Rank DU COOPs by mean Grid Value on the filtered frame.
    Excludes rows with null DU COOP or null Grid Value.
    """
    empty_worst = pd.DataFrame(columns=["DU COOP", "mean_grid", "n_site_months", "n_sites"])
    out: dict[str, Any] = {
        "best_names": [],
        "best_mean_grid": None,
        "best_site_months": 0,
        "best_sites": 0,
        "worst": empty_worst.copy(),
    }
    if df.empty or "DU COOP" not in df.columns or "Grid Value" not in df.columns:
        return out

    x = df.loc[df["DU COOP"].notna() & df["Grid Value"].notna()].copy()
    if x.empty:
        return out

    g = (
        x.groupby("DU COOP", dropna=False)
        .agg(mean_grid=("Grid Value", "mean"), n_site_months=("Grid Value", "size"), n_sites=("Site ID", "nunique"))
        .reset_index()
    )
    g = g.sort_values(["mean_grid", "DU COOP"], ascending=[False, True])
    max_mean = g["mean_grid"].max()
    best = g.loc[g["mean_grid"] == max_mean]
    best_keys = best["DU COOP"]
    mask = x["DU COOP"].isin(best_keys)
    out["best_names"] = best["DU COOP"].astype(str).tolist()
    out["best_mean_grid"] = float(max_mean) if pd.notna(max_mean) else None
    out["best_site_months"] = int(mask.sum())
    out["best_sites"] = int(x.loc[mask, "Site ID"].nunique())

    worst = g.sort_values(["mean_grid", "DU COOP"], ascending=[True, True]).head(worst_n)
    out["worst"] = worst[["DU COOP", "mean_grid", "n_site_months", "n_sites"]].reset_index(drop=True)
    return out


def best_du_coop_grid_monthly(df: pd.DataFrame, best_names: list[str]) -> pd.DataFrame:
    """
    Monthly mean Grid Value per DU COOP, restricted to best_names (e.g. tied top performers).
    Requires month_ts and month_label on df (normalized availability frame).
    """
    cols = ["month_ts", "month_label", "DU COOP", "Grid Value"]
    empty = pd.DataFrame(columns=cols)
    if df.empty or not best_names:
        return empty
    if "month_ts" not in df.columns or "month_label" not in df.columns:
        return empty
    sub = df.loc[df["DU COOP"].isin(best_names) & df["Grid Value"].notna()].copy()
    if sub.empty:
        return empty
    g = sub.groupby(["month_ts", "month_label", "DU COOP"], sort=True)["Grid Value"].mean().reset_index()
    return g.sort_values(["month_ts", "DU COOP"]).reset_index(drop=True)


def monthly_runtime_totals(d: pd.DataFrame) -> pd.DataFrame:
    """Average daily runtime hours per site-month by calendar month (mean across site-month rows)."""
    cols = RUNTIME_SOURCES + [TOTAL_RUNTIME_COL]
    if d.empty:
        return pd.DataFrame(columns=["month_ts", "month_label"] + cols)
    norm = _runtime_frame_for_averaging(d)
    g = norm.groupby(["month_ts", "month_label"], sort=True)[cols].mean().reset_index()
    return g.sort_values("month_ts")


def top_du_coops_by_dg_runtime(d: pd.DataFrame, *, n: int = 8) -> list[str]:
    """DU COOP keys with highest total DG_runtime over the frame."""
    if d.empty or "DU COOP" not in d.columns:
        return []
    s = (
        d.groupby("DU COOP", dropna=False)["DG_runtime"]
        .sum()
        .sort_values(ascending=False)
        .head(max(n, 0))
    )
    return [str(x) for x in s.index.tolist()]


def dim_mean_runtime_extremes(
    d: pd.DataFrame,
    dim: str,
    value_col: str,
    *,
    n: int = 5,
) -> tuple[list[str], list[str]]:
    """
    ``dim`` keys with highest and lowest mean ``value_col`` (daily hours / site-month row),
    after row normalization. Rows with null ``dim`` or null ``value_col`` are omitted;
    groups with no valid rows are dropped.
    """
    if d.empty or dim not in d.columns or value_col not in d.columns:
        return [], []
    norm = _runtime_frame_for_averaging(d)
    x = norm.loc[norm[dim].notna() & norm[value_col].notna()].copy()
    if x.empty:
        return [], []
    g = x.groupby(dim, dropna=False)[value_col].mean().dropna()
    if g.empty:
        return [], []
    means = g.reset_index()
    means.columns = [dim, "_mean"]
    kk = max(0, int(n))
    hi = (
        means.sort_values(["_mean", dim], ascending=[False, True])
        .head(kk)[dim]
        .astype(str)
        .tolist()
    )
    lo = (
        means.sort_values(["_mean", dim], ascending=[True, True])
        .head(kk)[dim]
        .astype(str)
        .tolist()
    )
    return hi, lo


def du_coop_mean_runtime_extremes(
    d: pd.DataFrame,
    coops: list[str],
    value_col: str,
    *,
    n: int = 5,
) -> tuple[list[str], list[str]]:
    """
    DU COOP keys with highest and lowest mean ``value_col`` (daily hours / site-month row),
    restricted to ``coops``, after row normalization. Rows with null ``value_col`` are omitted
    from that coop's mean; coops with no valid rows are dropped.
    """
    if d.empty or not coops or "DU COOP" not in d.columns or value_col not in d.columns:
        return [], []
    coop_set = {str(c) for c in coops}
    sub = d[d["DU COOP"].notna() & d["DU COOP"].astype(str).isin(coop_set)].copy()
    if sub.empty:
        return [], []
    return dim_mean_runtime_extremes(sub, "DU COOP", value_col, n=n)


def monthly_runtime_for_du_coop(d: pd.DataFrame, coop: str) -> pd.DataFrame:
    """Average daily runtime hours per month for one DU COOP."""
    sub = d[d["DU COOP"].astype(str) == str(coop)]
    return monthly_runtime_totals(sub)


def monthly_runtime_for_dim(d: pd.DataFrame, dim: str, keys: list[str]) -> pd.DataFrame:
    """Average daily runtime hours per site-month per ``dim``/month (mean across rows)."""
    cols = RUNTIME_SOURCES + [TOTAL_RUNTIME_COL]
    if d.empty or not keys or dim not in d.columns:
        return pd.DataFrame(columns=["month_ts", "month_label", dim] + cols)
    key_set = {str(k) for k in keys}
    sub = d[d[dim].notna() & d[dim].astype(str).isin(key_set)]
    if sub.empty:
        return pd.DataFrame(columns=["month_ts", "month_label", dim] + cols)
    norm = _runtime_frame_for_averaging(sub)
    g = norm.groupby(["month_ts", "month_label", dim], sort=True)[cols].mean().reset_index()
    return g.sort_values(["month_ts", dim])


def monthly_runtime_for_du_coops(d: pd.DataFrame, coops: list[str]) -> pd.DataFrame:
    """Average daily runtime hours per site-month per DU COOP/month (mean across rows)."""
    return monthly_runtime_for_dim(d, "DU COOP", coops)


def runtime_splits_by_dim(d: pd.DataFrame, dim: str) -> dict[str, pd.DataFrame]:
    """Map dim value -> monthly mean daily runtime metrics for that slice."""
    out: dict[str, pd.DataFrame] = {}
    if d.empty or dim not in d.columns:
        return out
    for key, g in d.groupby(dim, dropna=False):
        label = str(key) if pd.notna(key) else "(missing)"
        out[label] = monthly_runtime_totals(g)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def pivot_du_coop_month(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """DU COOP × month_label mean daily hours for heatmaps (normalized per row, mean across sites)."""
    if d.empty:
        return pd.DataFrame()
    norm = _runtime_frame_for_averaging(d)
    if value_col not in norm.columns:
        return pd.DataFrame()
    t = norm.groupby(["DU COOP", "month_label"], dropna=False)[value_col].mean().reset_index()
    wide = t.pivot(index="DU COOP", columns="month_label", values=value_col)
    month_order = sorted(wide.columns.astype(str))
    return wide[month_order].fillna(0.0)


def pivot_province_month(d: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Province × month_label mean daily hours (normalized per row, mean across sites)."""
    if d.empty or "Province" not in d.columns:
        return pd.DataFrame()
    norm = _runtime_frame_for_averaging(d)
    if value_col not in norm.columns:
        return pd.DataFrame()
    t = norm.groupby(["Province", "month_label"], dropna=False)[value_col].mean().reset_index()
    wide = t.pivot(index="Province", columns="month_label", values=value_col)
    month_order = sorted(wide.columns.astype(str))
    return wide[month_order].fillna(0.0)
