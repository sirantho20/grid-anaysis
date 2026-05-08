"""Plotly figure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {"Grid": "#2563eb", "DG": "#ea580c", "Battery": "#16a34a"}

_SOURCE_ORDER = ["Grid", "DG (genset)", "Battery"]


def _fmt2(values: object, *, suffix: str = "") -> list[str]:
    """Two-decimal display labels, preserving blanks for null/non-numeric values."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    return [f"{v:.2f}{suffix}" if pd.notna(v) else "" for v in vals]


def _fmt2_matrix(values: object, *, suffix: str = "") -> list[list[str]]:
    arr = np.asarray(values, dtype=float)
    labels = np.empty(arr.shape, dtype=object)
    finite = np.isfinite(arr)
    if finite.any():
        labels[finite] = np.vectorize(lambda v: f"{v:.2f}{suffix}")(arr[finite])
    labels[~finite] = ""
    return labels.tolist()


def _with_text_mode(mode: str | None) -> str:
    parts = set(str(mode or "markers").split("+"))
    parts.add("text")
    ordered = [p for p in ("lines", "markers", "text") if p in parts]
    return "+".join(ordered)


def _format_numeric_display(fig: go.Figure) -> go.Figure:
    """Apply visible two-decimal labels and rounded hover text to all numeric traces."""
    for trace in fig.data:
        trace_type = getattr(trace, "type", "")
        if trace_type == "bar":
            trace.update(
                text=_fmt2(trace.y),
                texttemplate="%{text}",
                textposition="auto",
                hovertemplate="%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        elif trace_type == "histogram":
            trace.update(
                texttemplate="%{y:.2f}",
                hovertemplate="%{x:.2f}<br>Count: %{y:.2f}<extra>%{fullData.name}</extra>",
            )
        elif trace_type == "heatmap":
            trace.update(
                text=_fmt2_matrix(trace.z),
                texttemplate="%{text}",
                hovertemplate=trace.hovertemplate or "%{y}<br>%{x}<br>%{z:.2f}<extra></extra>",
                colorbar_tickformat=".2f",
            )
        elif trace_type == "scatter" and getattr(trace, "y", None) is not None:
            trace.update(
                mode=_with_text_mode(trace.mode),
                text=_fmt2(trace.y),
                texttemplate="%{text}",
                textposition=getattr(trace, "textposition", None) or "top center",
                textfont=dict(size=10),
            )
            if not trace.hovertemplate:
                trace.update(hovertemplate="%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>")

    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".2f"))
    return fig


def _base_layout(fig: go.Figure, *, title: str) -> go.Figure:
    _format_numeric_display(fig)
    fig.update_layout(
        template="plotly_white",
        title=title,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        margin=dict(l=48, r=24, t=56, b=48),
        hovermode="x unified",
    )
    return fig


def fig_portfolio_lines(means: pd.DataFrame, *, subtitle: str = "") -> go.Figure:
    if means.empty:
        return _empty_fig("Portfolio trends (no data)")
    fig = go.Figure()
    x = means["month_ts"]
    fig.add_trace(go.Scatter(x=x, y=means["Grid Value"], name="Grid (mean %)", line=dict(color=PALETTE["Grid"])))
    fig.add_trace(go.Scatter(x=x, y=means["DG Value"], name="DG (mean %)", line=dict(color=PALETTE["DG"])))
    fig.add_trace(
        go.Scatter(x=x, y=means["Battery Value"], name="Battery (mean %)", line=dict(color=PALETTE["Battery"]))
    )
    t = "Mean site availability by month — Grid, DG, Battery (%)" + (f" — {subtitle}" if subtitle else "")
    fig.update_yaxes(title_text="Availability (%)", range=[0, 105])
    fig.update_xaxes(title_text="Month")
    return _base_layout(fig, title=t)


def fig_mix_shares(shares: pd.DataFrame) -> go.Figure:
    if shares.empty:
        return _empty_fig("Mix composition (no data)")
    x = shares["month_ts"]
    y1 = shares["Grid Value_share"] * 100
    y2 = shares["DG Value_share"] * 100
    y3 = shares["Battery Value_share"] * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            stackgroup="one",
            name="Grid share",
            fillcolor="rgba(37,99,235,0.35)",
            line=dict(width=0),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            stackgroup="one",
            name="DG share",
            fillcolor="rgba(234,88,12,0.35)",
            line=dict(width=0),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y3,
            stackgroup="one",
            name="Battery share",
            fillcolor="rgba(22,163,74,0.35)",
            line=dict(width=0),
            mode="lines",
        )
    )
    fig.update_yaxes(title_text="Share of Grid+DG+Battery (%)", range=[0, 100])
    fig.update_xaxes(title_text="Month")
    return _base_layout(fig, title="Average source mix (share of Grid+DG+Battery)")


def fig_mom_bars(means: pd.DataFrame) -> go.Figure:
    if means.empty or len(means) < 2:
        return _empty_fig("Month-over-month change (needs ≥2 months)")
    means = means.sort_values("month_ts")
    last = means.iloc[-1]
    prev = means.iloc[-2]
    cats = ["Grid", "DG", "Battery"]
    vals = [
        float(last["Grid Value"] - prev["Grid Value"]),
        float(last["DG Value"] - prev["DG Value"]),
        float(last["Battery Value"] - prev["Battery Value"]),
    ]
    cols = [PALETTE["Grid"], PALETTE["DG"], PALETTE["Battery"]]
    fig = go.Figure(go.Bar(x=cats, y=vals, marker_color=cols))
    fig.update_yaxes(title_text="Δ mean availability (percentage points)")
    return _base_layout(
        fig,
        title=f"Latest month vs prior ({prev['month_label']} → {last['month_label']})",
    )


def fig_component_histograms(d: pd.DataFrame) -> go.Figure:
    if d.empty:
        return _empty_fig("Distributions (no data)")
    colors = [PALETTE["Grid"], PALETTE["DG"], PALETTE["Battery"]]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Grid", "DG", "Battery"])
    for i, col in enumerate(["Grid Value", "DG Value", "Battery Value"], start=1):
        fig.add_trace(
            go.Histogram(x=d[col], nbinsx=40, name=col, marker_color=colors[i - 1], showlegend=False),
            row=1,
            col=i,
        )
    fig.update_xaxes(title_text="Availability (%)")
    return _base_layout(fig, title="Site-month distributions by source")


def fig_scatter_substitution(d: pd.DataFrame, *, max_points: int = 4000) -> go.Figure:
    if d.empty:
        return _empty_fig("Grid vs DG (no data)")
    x = d[["Grid Value", "DG Value", "Battery Value", "Site Name", "month_label", "DU COOP"]].dropna()
    x = x.copy()
    x["avg3"] = x[["Grid Value", "DG Value", "Battery Value"]].mean(axis=1)
    if len(x) > max_points:
        x = x.sample(max_points, random_state=42)
    fig = px.scatter(
        x,
        x="Grid Value",
        y="DG Value",
        color="avg3",
        hover_data={
            "Site Name": True,
            "month_label": True,
            "DU COOP": True,
            "Grid Value": ":.2f",
            "DG Value": ":.2f",
            "Battery Value": ":.2f",
            "avg3": ":.2f",
        },
        color_continuous_scale="Viridis",
        labels={"avg3": "Avg Grid/DG/Battery (%)"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.45))
    fig.update_xaxes(title_text="Grid availability (%)", tickformat=".2f")
    fig.update_yaxes(title_text="DG availability (%)")
    return _base_layout(fig, title="Substitution view: grid vs DG (sampled if large)")


def fig_correlation(corr: pd.DataFrame) -> go.Figure:
    if corr.empty:
        return _empty_fig("Correlation (no data)")
    labels = {"Grid Value": "Grid", "DG Value": "DG", "Battery Value": "Battery"}
    disp = corr.rename(index=labels, columns=labels)
    fig = px.imshow(
        disp,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(hovermode="closest")
    return _base_layout(fig, title="Correlation among Grid, DG, and Battery")


def fig_site_heatmaps(heat_grid: pd.DataFrame, heat_dg: pd.DataFrame, heat_batt: pd.DataFrame) -> go.Figure:
    if heat_grid.empty and heat_dg.empty and heat_batt.empty:
        return _empty_fig("Site heatmaps (no data)")

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Grid availability (%)", "DG / genset (%)", "Battery (%)"),
        vertical_spacing=0.12,
    )
    for row, heat, title_suffix in (
        (1, heat_grid, "Grid"),
        (2, heat_dg, "DG"),
        (3, heat_batt, "Battery"),
    ):
        if heat.empty:
            continue
        z = heat.to_numpy()
        lo = float(np.nanpercentile(z, 5)) if z.size else 0.0
        hi = float(np.nanpercentile(z, 95)) if z.size else 100.0
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=list(heat.columns),
                y=list(heat.index.astype(str)),
                colorscale="YlGnBu",
                zmin=lo,
                zmax=hi,
                showscale=row == 1,
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(autorange="reversed", title_text="Site ID")
    fig.update_xaxes(title_text="Month")
    fig.update_layout(height=900, hovermode="closest", margin=dict(t=80))
    return _base_layout(fig, title="Top volatile sites (by variability of avg Grid/DG/Battery)")


def fig_geo_bar(g_long: pd.DataFrame, dim: str, *, month_focus: str | None) -> go.Figure:
    if g_long.empty:
        return _empty_fig("Geographic breakdown (no data)")
    fig = px.bar(
        g_long,
        x=dim,
        y="availability",
        color="source",
        barmode="group",
        category_orders={"source": _SOURCE_ORDER},
        color_discrete_map={"Grid": PALETTE["Grid"], "DG (genset)": PALETTE["DG"], "Battery": PALETTE["Battery"]},
        labels={"availability": "Mean availability (%)", "source": "Source"},
    )
    fig.update_xaxes(tickangle=-35)
    suffix = f" — {month_focus}" if month_focus else " — filtered months"
    return _base_layout(fig, title=f"Mean availability by source and {dim}{suffix}")


def fig_best_du_coop_grid_lines(monthly: pd.DataFrame, *, subtitle: str = "") -> go.Figure:
    if monthly.empty:
        return _empty_fig("Best DU COOP(s) — mean Grid (no data)")
    fig = px.line(
        monthly,
        x="month_ts",
        y="Grid Value",
        color="DU COOP",
        markers=True,
        labels={"Grid Value": "Mean Grid (%)", "DU COOP": "DU COOP", "month_ts": "Month"},
    )
    fig.update_yaxes(title_text="Mean Grid (%)", range=[0, 105])
    fig.update_xaxes(title_text="Month")
    t = "Best DU COOP(s) — mean Grid availability by month" + (f" — {subtitle}" if subtitle else "")
    return _base_layout(fig, title=t)


def fig_du_coop_lines(long_df: pd.DataFrame) -> go.Figure:
    if long_df.empty:
        return _empty_fig("DU COOP trend (no data)")
    fig = px.line(
        long_df,
        x="month_ts",
        y="availability",
        color="DU COOP",
        facet_row="source",
        category_orders={"source": _SOURCE_ORDER},
        labels={"availability": "Mean (%)", "DU COOP": "DU COOP"},
    )
    fig.update_yaxes(title_text="Mean (%)")
    fig.update_xaxes(title_text="Month")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(hovermode="x unified", height=700, legend_traceorder="normal")
    return _base_layout(fig, title="Mean availability by DU COOP — Grid, DG, Battery (top by site count)")


def fig_monthly_runtime_sources_and_total(totals: pd.DataFrame, *, title: str, subtitle: str = "") -> go.Figure:
    """Grouped bars for average daily Grid/DG/Battery hours with Total_runtime line on the same Y-axis."""
    if totals.empty:
        return _empty_fig(title)
    x = totals["month_ts"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=totals["Grid_runtime"], name="Grid (h/day)", marker_color=PALETTE["Grid"]))
    fig.add_trace(go.Bar(x=x, y=totals["DG_runtime"], name="DG (h/day)", marker_color=PALETTE["DG"]))
    fig.add_trace(go.Bar(x=x, y=totals["Battery_runtime"], name="Battery (h/day)", marker_color=PALETTE["Battery"]))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=totals["Total_runtime"],
            name="Total (h/day)",
            mode="lines+markers",
            line=dict(color="#64748b", width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Average hours / day")
    fig.update_xaxes(title_text="Month")
    full_title = title + (f" — {subtitle}" if subtitle else "")
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, title=full_title)


def fig_portfolio_runtime_monthly(totals: pd.DataFrame, *, subtitle: str = "") -> go.Figure:
    return fig_monthly_runtime_sources_and_total(
        totals,
        title="Portfolio — average daily runtime hours per site-month",
        subtitle=subtitle,
    )


def fig_du_coop_runtime_single(totals: pd.DataFrame, *, coop: str) -> go.Figure:
    sub = f"DU COOP: {coop}"
    return fig_monthly_runtime_sources_and_total(
        totals,
        title="DU COOP — average daily runtime hours per site-month",
        subtitle=sub,
    )


def fig_du_coop_runtime_lines_for_metric(
    monthly_by_coop: pd.DataFrame,
    *,
    coops: list[str],
    value_col: str,
    y_title: str,
    title: str,
    group_col: str = "DU COOP",
) -> go.Figure:
    """One metric, one line per series key in ``group_col`` (separate figures — avoids subplot chrome)."""
    if monthly_by_coop.empty or not coops or group_col not in monthly_by_coop.columns:
        return _empty_fig(title)
    coop_colors = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, coop in enumerate(coops):
        sub = monthly_by_coop[monthly_by_coop[group_col].astype(str) == str(coop)]
        color = coop_colors[i % len(coop_colors)]
        fig.add_trace(
            go.Scatter(
                x=sub["month_ts"],
                y=sub[value_col],
                mode="lines+markers",
                name=str(coop),
                line=dict(width=2, color=color),
                marker=dict(size=6),
            )
        )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(hovermode="x unified", height=420)
    return _base_layout(fig, title=title)


def fig_runtime_heatmap(matrix: pd.DataFrame, *, title: str, y_axis_title: str) -> go.Figure:
    if matrix.empty:
        return _empty_fig(title)
    z = matrix.to_numpy(dtype=float)
    x = [str(c) for c in matrix.columns]
    y = [str(i) for i in matrix.index]
    note = ""
    flat = z[np.isfinite(z)]
    if flat.size:
        p90 = float(np.percentile(flat, 90))
        note = f" · cell values ≥ 90th percentile ({p90:.2f} h/day) are highest-intensity"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Reds",
            hovertemplate="%{y}<br>%{x}<br>%{z:.2f} h/day<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title_text=y_axis_title)
    fig.update_xaxes(title_text="Month")
    fig.update_layout(hovermode="closest", margin=dict(t=56))
    return _base_layout(fig, title=title + note)


def fig_nulls(null_rates: dict) -> go.Figure:
    if not null_rates:
        return _empty_fig("Null rates")
    items = sorted(null_rates.items(), key=lambda kv: kv[1], reverse=True)
    cols = [k for k, _ in items]
    vals = [v * 100 for _, v in items]
    fig = go.Figure(go.Bar(x=cols, y=vals, marker_color="#64748b"))
    fig.update_yaxes(title_text="Null rate (%)")
    fig.update_xaxes(tickangle=-30)
    return _base_layout(fig, title="Missingness for key columns")


def fig_reconcile_trend(monthly: pd.DataFrame) -> go.Figure:
    if monthly.empty:
        return _empty_fig("Reconciliation trend")
    fig = go.Figure(
        go.Scatter(
            x=pd.to_datetime(monthly["month_label"] + "-01"),
            y=monthly["fail_rate"] * 100,
            mode="lines+markers",
            name="Fail rate",
        )
    )
    fig.update_yaxes(title_text="Rows failing sum check (%)")
    fig.update_xaxes(title_text="Month")
    return _base_layout(fig, title="Share of rows where |Grid+DG+Battery−Total| exceeds tolerance")


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        annotations=[
            dict(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        ],
    )
    return fig
