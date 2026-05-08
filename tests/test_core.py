import pandas as pd

from data.load import _normalize_frame, parse_month
from data.quality import reconciliation_delta, run_quality_report
from metrics import aggregate as ag
from metrics import figures as figs


def test_parse_month():
    s = pd.Series(["2025-01", "2025-12", "bad"])
    out = parse_month(s)
    assert pd.isna(out.iloc[2])
    assert out.iloc[0] == pd.Timestamp("2025-01-01")


def test_normalize_adds_month_columns():
    df = pd.DataFrame(
        {
            "Month": ["2025-01"],
            "Site ID": ["S1"],
            "Site Name": ["A"],
            "Globe ID": ["G"],
            "Portfolio": ["P"],
            "Power_Model": ["Grid + Battery"],
            "Indoor/Outdoor": ["OUTDOOR"],
            "DU COOP": ["X"],
            "Territory": ["T"],
            "District": ["D"],
            "Province": ["P"],
            "Area": ["A"],
            "RMS Type": ["R"],
            "Toggle": ["Availability"],
            "Grid Value": [50.0],
            "DG Value": [30.0],
            "Battery Value": [20.0],
            "Total Value": [100.0],
        }
    )
    out = _normalize_frame(df)
    assert "month_ts" in out.columns
    assert out["month_label"].iloc[0] == "2025-01"


def test_reconciliation_delta_perfect():
    df = pd.DataFrame(
        {
            "Grid Value": [60.0, 10.0],
            "DG Value": [30.0, 50.0],
            "Battery Value": [10.0, 40.0],
            "Total Value": [100.0, 100.0],
        }
    )
    d = reconciliation_delta(df)
    assert (d < 1e-9).all()


def test_quality_flags_bad_sum():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01"],
            "Site ID": ["a"],
            "Grid Value": [30.0],
            "DG Value": [30.0],
            "Battery Value": [30.0],
            "Total Value": [100.0],
        }
    )
    rep = run_quality_report(df, reconcile_atol=0.5, reconcile_rtol=0.0)
    assert rep["n_reconcile_failures"] == 1


def test_apply_filters_months():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-02"],
            "month_ts": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "Site ID": ["x", "x"],
            "Site Name": ["n", "n"],
            "Globe ID": ["g", "g"],
            "Portfolio": ["p", "p"],
            "Power_Model": ["pm", "pm"],
            "Indoor/Outdoor": ["o", "o"],
            "DU COOP": ["d", "d"],
            "Territory": ["t", "t"],
            "District": ["di", "di"],
            "Province": ["pr", "pr"],
            "Area": ["a", "a"],
            "RMS Type": ["r", "r"],
            "Toggle": ["v", "v"],
            "Grid Value": [90.0, 91.0],
            "DG Value": [5.0, 4.0],
            "Battery Value": [5.0, 5.0],
            "Total Value": [100.0, 100.0],
        }
    )
    out = ag.apply_filters(df, ["2025-02"], None, None, None, None, None)
    assert len(out) == 1


def test_mix_shares_row_fractions_sum_to_one():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-01"],
            "month_ts": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "Grid Value": [30.0, 10.0],
            "DG Value": [50.0, 60.0],
            "Battery Value": [20.0, 30.0],
            "Total Value": [100.0, 100.0],
        }
    )
    sh = ag.mix_shares_by_month(df)
    assert len(sh) == 1
    row = sh.iloc[0]
    assert abs(row["Grid Value_share"] + row["DG Value_share"] + row["Battery Value_share"] - 1.0) < 1e-9


def test_portfolio_means_groups():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01"] * 2,
            "month_ts": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "Grid Value": [80.0, 60.0],
            "DG Value": [10.0, 20.0],
            "Battery Value": [10.0, 20.0],
            "Total Value": [100.0, 100.0],
        }
    )
    m = ag.portfolio_means_by_month(df)
    assert abs(m["Grid Value"].iloc[0] - 70.0) < 1e-9


def test_du_coop_grid_rankings_best_worst_and_tie():
    df = pd.DataFrame(
        {
            "DU COOP": ["HighCoop", "HighCoop", "LowCoop", "LowCoop", "Tie1", "Tie2"],
            "Site ID": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "Grid Value": [80.0, 80.0, 20.0, 40.0, 95.0, 95.0],
        }
    )
    r = ag.du_coop_grid_rankings(df, worst_n=5)
    assert set(r["best_names"]) == {"Tie1", "Tie2"}
    assert r["best_mean_grid"] == 95.0
    assert r["best_site_months"] == 2
    assert r["best_sites"] == 2
    worst = r["worst"]
    assert worst.iloc[0]["DU COOP"] == "LowCoop"
    assert worst.iloc[0]["mean_grid"] == 30.0
    assert worst.iloc[1]["DU COOP"] == "HighCoop"
    assert worst.iloc[1]["mean_grid"] == 80.0


def test_du_coop_grid_rankings_excludes_nulls():
    df = pd.DataFrame(
        {
            "DU COOP": ["A", None, "B"],
            "Site ID": ["s1", "s2", "s3"],
            "Grid Value": [50.0, 99.0, 10.0],
        }
    )
    r = ag.du_coop_grid_rankings(df)
    assert r["best_names"] == ["A"]
    assert r["worst"].iloc[0]["DU COOP"] == "B"
    assert r["worst"].iloc[0]["mean_grid"] == 10.0


def test_du_coop_grid_rankings_empty():
    r = ag.du_coop_grid_rankings(pd.DataFrame())
    assert r["best_names"] == []
    assert r["worst"].empty


def test_best_du_coop_grid_monthly():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-01", "2025-02", "2025-02", "2025-01"],
            "month_ts": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01", "2025-01-01"]
            ),
            "DU COOP": ["BestCoop", "BestCoop", "BestCoop", "BestCoop", "Other"],
            "Site ID": ["a", "b", "a", "b", "c"],
            "Grid Value": [80.0, 90.0, 60.0, 80.0, 99.0],
        }
    )
    m = ag.best_du_coop_grid_monthly(df, ["BestCoop"])
    assert len(m) == 2
    jan = m[m["month_label"] == "2025-01"].iloc[0]
    feb = m[m["month_label"] == "2025-02"].iloc[0]
    assert jan["Grid Value"] == 85.0
    assert feb["Grid Value"] == 70.0
    assert m["DU COOP"].eq("BestCoop").all()


def test_monthly_runtime_totals():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-01", "2025-02"],
            "month_ts": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01"]),
            "DU COOP": ["A", "B", "A"],
            "Area": ["X", "X", "X"],
            "Territory": ["T", "T", "T"],
            "Province": ["P", "P", "P"],
            "Grid_runtime": [10.0, 5.0, 3.0],
            "DG_runtime": [1.0, 20.0, 2.0],
            "Battery_runtime": [0.0, 0.0, 1.0],
            "Total_runtime": [11.0, 25.0, 6.0],
        }
    )
    m = ag.monthly_runtime_totals(df)
    assert len(m) == 2
    jan = m[m["month_label"] == "2025-01"].iloc[0]
    feb = m[m["month_label"] == "2025-02"].iloc[0]
    # Jan: second row DG+Grid scaled to Total cap 24; then column means across two sites
    assert jan["Grid_runtime"] == (10.0 + 5.0 * 24.0 / 25.0) / 2.0
    assert jan["DG_runtime"] == (1.0 + 20.0 * 24.0 / 25.0) / 2.0
    assert jan["Battery_runtime"] == 0.0
    assert jan["Total_runtime"] == (11.0 + 24.0) / 2.0
    assert feb["Grid_runtime"] == 3.0
    assert feb["DG_runtime"] == 2.0
    assert feb["Battery_runtime"] == 1.0
    assert feb["Total_runtime"] == 6.0


def test_monthly_runtime_combined_sources_within_24h_after_aggregate():
    """Sources scaled to Total cap row-wise; monthly mean(Grid+DG+Battery) stays <= 24."""
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-01"],
            "month_ts": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "DU COOP": ["A", "A"],
            "Grid_runtime": [24.0, 0.0],
            "DG_runtime": [12.0, 24.0],
            "Battery_runtime": [0.0, 0.0],
            "Total_runtime": [24.0, 24.0],
        }
    )
    m = ag.monthly_runtime_totals(df)
    assert len(m) == 1
    row = m.iloc[0]
    combined = float(row["Grid_runtime"] + row["DG_runtime"] + row["Battery_runtime"])
    assert combined <= 24.000001
    assert abs(combined - 24.0) < 1e-9


def test_top_du_coops_by_dg_runtime_orders_and_truncates():
    df = pd.DataFrame(
        {
            "DU COOP": ["Low", "High", "Mid", "High"],
            "DG_runtime": [1.0, 50.0, 10.0, 5.0],
        }
    )
    top2 = ag.top_du_coops_by_dg_runtime(df, n=2)
    assert top2 == ["High", "Mid"]
    top_all = ag.top_du_coops_by_dg_runtime(df, n=10)
    assert top_all == ["High", "Mid", "Low"]


def test_du_coop_mean_runtime_extremes_orders_by_mean_and_tiebreak():
    df = pd.DataFrame(
        {
            "DU COOP": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "month_label": ["2025-01"] * 8,
            "month_ts": pd.to_datetime(["2025-01-01"] * 8),
            "Grid_runtime": [10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0],
            "DG_runtime": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0],
            "Battery_runtime": [0.0] * 8,
            "Total_runtime": [11.0, 11.0, 7.0, 7.0, 4.0, 4.0, 7.0, 7.0],
        }
    )
    coops = ["A", "B", "C", "D"]
    hi, lo = ag.du_coop_mean_runtime_extremes(df, coops, "Grid_runtime", n=2)
    assert hi == ["A", "B"]
    assert lo == ["C", "B"]
    hi_dg, lo_dg = ag.du_coop_mean_runtime_extremes(df, coops, "DG_runtime", n=2)
    assert hi_dg == ["C", "B"]
    assert lo_dg == ["A", "B"]


def test_dim_mean_runtime_extremes_area_and_monthly_for_dim():
    areas = [f"R{i}" for i in range(1, 7)]
    df = pd.DataFrame(
        {
            "Area": areas,
            "month_label": ["2025-01"] * 6,
            "month_ts": pd.to_datetime(["2025-01-01"] * 6),
            "Site ID": [f"S{i}" for i in range(1, 7)],
            "DU COOP": ["X"] * 6,
            "Grid_runtime": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "DG_runtime": [0.0] * 6,
            "Battery_runtime": [0.0] * 6,
            "Total_runtime": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    hi, lo = ag.dim_mean_runtime_extremes(df, "Area", "Grid_runtime", n=5)
    assert hi == ["R6", "R5", "R4", "R3", "R2"]
    assert lo == ["R1", "R2", "R3", "R4", "R5"]

    m = ag.monthly_runtime_for_dim(df, "Area", ["R1", "R6"])
    assert len(m) == 2
    assert set(m["Area"].tolist()) == {"R1", "R6"}
    r1 = m[m["Area"] == "R1"].iloc[0]
    r6 = m[m["Area"] == "R6"].iloc[0]
    assert r1["Grid_runtime"] == 1.0
    assert r6["Grid_runtime"] == 6.0


def test_monthly_runtime_for_du_coop_and_splits():
    df = pd.DataFrame(
        {
            "month_label": ["2025-01", "2025-01"],
            "month_ts": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "DU COOP": ["X", "Y"],
            "Area": ["R1", "R2"],
            "Territory": ["Ta", "Tb"],
            "Province": ["Pa", "Pb"],
            "Grid_runtime": [4.0, 6.0],
            "DG_runtime": [1.0, 2.0],
            "Battery_runtime": [0.5, 0.5],
            "Total_runtime": [5.5, 8.5],
        }
    )
    one = ag.monthly_runtime_for_du_coop(df, "X")
    assert len(one) == 1
    assert one["Grid_runtime"].iloc[0] == 4.0

    multi = ag.monthly_runtime_for_du_coops(df, ["X", "Y"])
    assert len(multi) == 2

    by_area = ag.runtime_splits_by_dim(df, "Area")
    assert set(by_area.keys()) == {"R1", "R2"}
    assert by_area["R1"]["Grid_runtime"].iloc[0] == 4.0


def test_pivot_du_coop_and_province():
    df = pd.DataFrame(
        {
            "DU COOP": ["A", "A", "B"],
            "Province": ["P1", "P1", "P2"],
            "month_label": ["2025-01", "2025-02", "2025-01"],
            "DG_runtime": [2.0, 3.0, 10.0],
            "Grid_runtime": [1.0, 1.0, 5.0],
            "Battery_runtime": [0.0, 0.0, 0.0],
            "Total_runtime": [3.0, 4.0, 15.0],
        }
    )
    d_mat = ag.pivot_du_coop_month(df, "DG_runtime")
    assert d_mat.loc["A", "2025-01"] == 2.0
    assert d_mat.loc["A", "2025-02"] == 3.0
    assert d_mat.loc["B", "2025-01"] == 10.0

    p_mat = ag.pivot_province_month(df, "Grid_runtime")
    assert p_mat.loc["P1", "2025-01"] == 1.0
    assert p_mat.loc["P2", "2025-01"] == 5.0


def test_runtime_figure_has_two_decimal_data_labels():
    df = pd.DataFrame(
        {
            "month_ts": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "Grid_runtime": [23.123, 22.4],
            "DG_runtime": [0.345, 0.678],
            "Battery_runtime": [0.111, 0.222],
            "Total_runtime": [23.579, 23.3],
        }
    )
    fig = figs.fig_portfolio_runtime_monthly(df)

    assert fig.data[0].text[0] == "23.12"
    assert fig.data[0].texttemplate == "%{text}"
    assert "text" in fig.data[-1].mode
    assert fig.data[-1].text[0] == "23.58"


def test_heatmap_figure_has_two_decimal_cell_labels():
    fig = figs.fig_runtime_heatmap(
        pd.DataFrame([[1.234, 2.0]], index=["A"], columns=["2025-01", "2025-02"]),
        title="Heat",
        y_axis_title="DU COOP",
    )

    assert fig.data[0].text == (["1.23", "2.00"],)
    assert fig.data[0].texttemplate == "%{text}"
