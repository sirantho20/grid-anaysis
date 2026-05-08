"""
Runtime dashboard — Combined Data from grid_data2.xlsx.

Run from project root (with venv activated):
  python app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import os
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, dcc, html

from data.load import load_combined
from metrics import aggregate as ag
from metrics import figures as figs

_DATA: pd.DataFrame | None = None


def get_data() -> pd.DataFrame:
    global _DATA
    if _DATA is None:
        _DATA = load_combined(use_cache=True)
    return _DATA


def _subtitle(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows"
    months = sorted(df["month_label"].dropna().unique().tolist())
    if not months:
        return f"{len(df):,} rows"
    return f"{len(df):,} site-months · {months[0]}–{months[-1]}"


def _du_coop_extreme_runtime_charts(
    df: pd.DataFrame, active_coops: list[str], title_suffix: str
) -> list[Any]:
    """Top / lowest n DU COOPs by mean daily runtime per source (Grid, Genset, Battery)."""
    out: list[Any] = []
    sources: tuple[tuple[str, str], ...] = (
        ("Grid", "Grid_runtime"),
        ("Genset", "DG_runtime"),
        ("Battery", "Battery_runtime"),
    )
    for label, value_col in sources:
        hi, lo = ag.du_coop_mean_runtime_extremes(df, active_coops, value_col, n=5)
        if hi:
            m_hi = ag.monthly_runtime_for_du_coops(df, hi)
            out.append(html.H6(f"{label} — top 5 by mean runtime", className="mt-3 mb-2"))
            out.append(
                dcc.Graph(
                    figure=figs.fig_du_coop_runtime_lines_for_metric(
                        m_hi,
                        coops=hi,
                        value_col=value_col,
                        y_title="Avg hours / day",
                        title=f"{label} — top 5 DU COOPs by mean daily runtime ({title_suffix})",
                    ),
                    className="chart-xl mb-4",
                )
            )
        if lo:
            m_lo = ag.monthly_runtime_for_du_coops(df, lo)
            out.append(html.H6(f"{label} — lowest 5 by mean runtime", className="mb-2"))
            out.append(
                dcc.Graph(
                    figure=figs.fig_du_coop_runtime_lines_for_metric(
                        m_lo,
                        coops=lo,
                        value_col=value_col,
                        y_title="Avg hours / day",
                        title=f"{label} — lowest 5 DU COOPs by mean daily runtime ({title_suffix})",
                    ),
                    className="chart-xl mb-4",
                )
            )
    return out


def _area_extreme_runtime_charts(df: pd.DataFrame) -> list[Any]:
    """Top / lowest n Areas by mean daily runtime per source (Grid, Genset, Battery)."""
    out: list[Any] = []
    sources: tuple[tuple[str, str], ...] = (
        ("Grid", "Grid_runtime"),
        ("Genset", "DG_runtime"),
        ("Battery", "Battery_runtime"),
    )
    for label, value_col in sources:
        hi, lo = ag.dim_mean_runtime_extremes(df, "Area", value_col, n=5)
        if hi:
            m_hi = ag.monthly_runtime_for_dim(df, "Area", hi)
            out.append(html.H6(f"{label} — top 5 Areas by mean runtime", className="mt-3 mb-2"))
            out.append(
                dcc.Graph(
                    figure=figs.fig_du_coop_runtime_lines_for_metric(
                        m_hi,
                        coops=hi,
                        value_col=value_col,
                        y_title="Avg hours / day",
                        title=f"{label} — top 5 Areas by mean daily runtime",
                        group_col="Area",
                    ),
                    className="chart-xl mb-4",
                )
            )
        if lo:
            m_lo = ag.monthly_runtime_for_dim(df, "Area", lo)
            out.append(html.H6(f"{label} — lowest 5 Areas by mean runtime", className="mb-2"))
            out.append(
                dcc.Graph(
                    figure=figs.fig_du_coop_runtime_lines_for_metric(
                        m_lo,
                        coops=lo,
                        value_col=value_col,
                        y_title="Avg hours / day",
                        title=f"{label} — lowest 5 Areas by mean daily runtime",
                        group_col="Area",
                    ),
                    className="chart-xl mb-4",
                )
            )
    return out


def _runtime_chart_cards(df_by_label: dict[str, pd.DataFrame], heading: str) -> list[Any]:
    """dbc.Card + Graph per geographic slice."""
    cards: list[Any] = []
    if not df_by_label:
        return [html.P("No data.", className="text-muted")]
    for label, totals in df_by_label.items():
        figure = figs.fig_monthly_runtime_sources_and_total(
            totals,
            title=f"{heading}: {label}",
        )
        cards.append(
            dbc.Card(
                dbc.CardBody([dcc.Graph(figure=figure, className="chart-xl")]),
                className="shadow-sm mb-4",
            )
        )
    return cards


def serve_layout() -> Any:
    df0 = get_data()
    subt = _subtitle(df0)
    coop_vals = sorted({str(x) for x in df0["DU COOP"].dropna().unique().tolist()})
    du_options = [{"label": c, "value": c} for c in coop_vals]

    return dbc.Container(
        [
            html.Div(
                [
                    html.H1("Runtime dashboard", className="dashboard-title mb-1"),
                    html.P(subt, className="text-muted"),
                ],
                className="py-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("DU COOP focus (average daily hours per calendar month)", className="mb-3"),
                        dbc.Label("DU COOP"),
                        dcc.Dropdown(
                            id="du-coop-runtime",
                            options=du_options,
                            value=[],
                            multi=True,
                            placeholder="All DU COOPs — choose one or more to narrow",
                            clearable=True,
                            className="mb-0",
                        ),
                    ]
                ),
                className="filter-card mb-4 shadow-sm",
            ),
            html.H4("Portfolio — average daily runtime hours", className="mt-2 mb-3"),
            dcc.Graph(id="fig-portfolio-runtime", className="chart-xl mb-4"),
            html.H4("DU COOP — average daily runtime hours", className="mb-3"),
            html.Div(id="du-runtime-charts"),
            html.H4("By Area (region)", className="mt-4 mb-3"),
            html.Div(id="area-runtime-charts"),
            html.H4("By Territory", className="mt-4 mb-3"),
            html.Div(id="territory-runtime-charts"),
            html.H4("DG runtime — DU COOP × month (avg hours / day)", className="mt-4 mb-3"),
            dcc.Graph(id="fig-heatmap-dg-du", className="chart-xl mb-4"),
            html.H4("Province × month — DG and Grid runtime (avg hours / day)", className="mt-4 mb-3"),
            html.Div(id="province-heatmap-charts", className="mb-5"),
        ],
        fluid=True,
        className="pb-5",
    )


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Grid runtime",
)
app.layout = serve_layout


@app.callback(
    Output("fig-portfolio-runtime", "figure"),
    Output("du-runtime-charts", "children"),
    Output("area-runtime-charts", "children"),
    Output("territory-runtime-charts", "children"),
    Output("fig-heatmap-dg-du", "figure"),
    Output("province-heatmap-charts", "children"),
    Input("du-coop-runtime", "value"),
)
def refresh_runtime(du_sel: list[str] | None):
    df = get_data()
    sub = _subtitle(df)

    portfolio = ag.monthly_runtime_totals(df)
    fig_portfolio = figs.fig_portfolio_runtime_monthly(portfolio, subtitle=sub)

    all_coops = sorted({str(x) for x in df["DU COOP"].dropna().unique().tolist()})
    active_coops = sorted(du_sel) if du_sel else all_coops
    is_all = bool(
        active_coops
        and len(all_coops) == len(active_coops)
        and set(active_coops) == set(all_coops)
    )

    if not active_coops:
        du_children = [html.P("No DU COOP values in the data.", className="text-muted")]
    elif len(active_coops) == 1:
        one = ag.monthly_runtime_for_du_coop(df, str(active_coops[0]))
        du_children = [
            dcc.Graph(
                figure=figs.fig_du_coop_runtime_single(one, coop=str(active_coops[0])),
                className="chart-xl mb-4",
            )
        ]
    elif is_all:
        du_children = [
            html.P(
                "Full DU COOP set — use the filter above to compare a subset on one chart per source.",
                className="text-muted mb-3",
            ),
            html.H5(
                "Top / lowest 5 DU COOPs by mean daily runtime (each source) — full dataset",
                className="text-muted mb-3",
            ),
            *_du_coop_extreme_runtime_charts(df, active_coops, "all DU COOPs"),
        ]
    else:
        multi = ag.monthly_runtime_for_du_coops(df, active_coops)
        n = len(active_coops)
        heading = f"Selected DU COOPs (n={n}) — separate charts per source (avg hours / day)"
        title_suffix = f"selected DU COOPs (n={n})"
        du_children = [
            html.H5(heading, className="text-muted mb-3"),
            html.H6("Grid runtime", className="mt-2 mb-2"),
            dcc.Graph(
                figure=figs.fig_du_coop_runtime_lines_for_metric(
                    multi,
                    coops=active_coops,
                    value_col="Grid_runtime",
                    y_title="Avg hours / day",
                    title=f"Grid — avg daily ({title_suffix})",
                ),
                className="chart-xl mb-4",
            ),
            html.H6("DG runtime", className="mb-2"),
            dcc.Graph(
                figure=figs.fig_du_coop_runtime_lines_for_metric(
                    multi,
                    coops=active_coops,
                    value_col="DG_runtime",
                    y_title="Avg hours / day",
                    title=f"DG — avg daily ({title_suffix})",
                ),
                className="chart-xl mb-4",
            ),
            html.H6("Battery runtime", className="mb-2"),
            dcc.Graph(
                figure=figs.fig_du_coop_runtime_lines_for_metric(
                    multi,
                    coops=active_coops,
                    value_col="Battery_runtime",
                    y_title="Avg hours / day",
                    title=f"Battery — avg daily ({title_suffix})",
                ),
                className="chart-xl mb-4",
            ),
            html.H5(
                "Top / lowest 5 DU COOPs by mean daily runtime (within current selection)",
                className="text-muted mb-3 mt-4",
            ),
            *_du_coop_extreme_runtime_charts(df, active_coops, title_suffix),
        ]

    areas = ag.runtime_splits_by_dim(df, "Area")
    territories = ag.runtime_splits_by_dim(df, "Territory")
    area_children = (
        [
            html.H5(
                "Top / lowest 5 Areas by mean daily runtime (each source)",
                className="mb-3",
            ),
            *_area_extreme_runtime_charts(df),
        ]
        + _runtime_chart_cards(areas, "Area")
    )
    territory_children = _runtime_chart_cards(territories, "Territory")

    dg_du_pivot = ag.pivot_du_coop_month(df, "DG_runtime")
    fig_dg_du = figs.fig_runtime_heatmap(
        dg_du_pivot,
        title="DG runtime avg (hours / day) — DU COOP × month",
        y_axis_title="DU COOP",
    )

    dg_prov = ag.pivot_province_month(df, "DG_runtime")
    grid_prov = ag.pivot_province_month(df, "Grid_runtime")
    fig_dg_prov = figs.fig_runtime_heatmap(
        dg_prov,
        title="DG runtime avg (hours / day) — Province × month",
        y_axis_title="Province",
    )
    fig_grid_prov = figs.fig_runtime_heatmap(
        grid_prov,
        title="Grid runtime avg (hours / day) — Province × month",
        y_axis_title="Province",
    )
    prov_children: list[Any] = [
        dcc.Graph(figure=fig_dg_prov, className="chart-xl mb-4"),
        dcc.Graph(figure=fig_grid_prov, className="chart-xl mb-4"),
    ]

    return fig_portfolio, du_children, area_children, territory_children, fig_dg_du, prov_children


if __name__ == "__main__":
    get_data()
    app.run(
        debug=False,
        host=os.environ.get("DASH_HOST", "127.0.0.1"),
        port=int(os.environ.get("DASH_PORT", "8050")),
    )
