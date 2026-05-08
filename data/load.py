"""Load Availability sheet with optional Parquet cache."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = PROJECT_ROOT / "grid_data.xlsx"
CACHE_PATH = PROJECT_ROOT / "cache" / "availability.parquet"

DEFAULT_COMBINED_XLSX = PROJECT_ROOT / "grid_data2.xlsx"
COMBINED_CACHE_PATH = PROJECT_ROOT / "cache" / "combined.parquet"

SHEET_NAME = "Availability from Jan2025"
COMBINED_SHEET_NAME = "Combined Data"

# Columns as produced by pandas after skiprows=[0], header=0 on Combined Data sheet.
COMBINED_READ_COLUMNS = [
    "Month",
    "Site ID",
    "Site Name",
    "Globe ID",
    "Portfolio",
    "Power_Model",
    "Indoor/Outdoor",
    "DU COOP",
    "Territory",
    "District",
    "Province",
    "Area",
    "RMS Type",
    "Toggle",
    "Grid Value",
    "DG Value",
    "Battery Value",
    "Total Value",
    "Month.1",
    "Site ID.1",
    "Toggle.1",
    "Grid_runtime",
    "DG_runtime",
    "Battery_runtime",
    "Total_runtime",
]

RUNTIME_COLUMNS = ["Grid_runtime", "DG_runtime", "Battery_runtime", "Total_runtime"]

REQUIRED_COLUMNS = [
    "Month",
    "Site ID",
    "Site Name",
    "Globe ID",
    "Portfolio",
    "Power_Model",
    "Indoor/Outdoor",
    "DU COOP",
    "Territory",
    "District",
    "Province",
    "Area",
    "RMS Type",
    "Toggle",
    "Grid Value",
    "DG Value",
    "Battery Value",
    "Total Value",
]

VALUE_COLUMNS = ["Grid Value", "DG Value", "Battery Value", "Total Value"]
DIMENSION_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in VALUE_COLUMNS and c != "Month"]

COMBINED_STRING_DIMS = [
    "Site ID",
    "Site Name",
    "Globe ID",
    "Portfolio",
    "Power_Model",
    "Indoor/Outdoor",
    "DU COOP",
    "Territory",
    "District",
    "Province",
    "Area",
    "RMS Type",
    "Toggle",
]


def parse_month(series: pd.Series) -> pd.Series:
    """Parse YYYY-MM strings to datetime (first day of month)."""
    s = series.astype(str).str.strip()
    return pd.to_datetime(s + "-01", format="%Y-%m-%d", errors="coerce")


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in DIMENSION_COLUMNS:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().replace({"nan": pd.NA})
    for c in VALUE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["month_ts"] = parse_month(out["Month"])
    out = out.dropna(subset=["month_ts"])
    out["month_label"] = out["month_ts"].dt.strftime("%Y-%m")
    return out


def load_availability(
    xlsx_path: str | os.PathLike[str] | None = None,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load availability data from Excel or newer Parquet cache."""
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_XLSX
    if not path.is_file():
        raise FileNotFoundError(f"Excel not found: {path}")

    if use_cache and not force_refresh and CACHE_PATH.is_file():
        if CACHE_PATH.stat().st_mtime >= path.stat().st_mtime:
            df = pd.read_parquet(CACHE_PATH)
            _validate_schema(df)
            return df

    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
        usecols=REQUIRED_COLUMNS,
    )
    _validate_schema(df)
    df = _normalize_frame(df)

    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


COMBINED_NORMALIZED_COLUMNS = [
    "Month",
    "month_ts",
    "month_label",
    "Site ID",
    "Site Name",
    "Globe ID",
    "Portfolio",
    "Power_Model",
    "Indoor/Outdoor",
    "DU COOP",
    "Territory",
    "District",
    "Province",
    "Area",
    "RMS Type",
    "Toggle",
    "Grid Value",
    "DG Value",
    "Battery Value",
    "Total Value",
    *RUNTIME_COLUMNS,
]


def _validate_combined_schema(df: pd.DataFrame) -> None:
    missing = [c for c in COMBINED_READ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing combined columns: {missing}")


def _validate_combined_normalized(df: pd.DataFrame) -> None:
    missing = [c for c in COMBINED_NORMALIZED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing normalized combined columns: {missing}")


def _normalize_combined_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_extra = [c for c in ("Month.1", "Site ID.1", "Toggle.1") if c in out.columns]
    out = out.drop(columns=drop_extra, errors="ignore")
    for c in COMBINED_STRING_DIMS:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().replace({"nan": pd.NA})
    for c in VALUE_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in RUNTIME_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["month_ts"] = parse_month(out["Month"])
    out = out.dropna(subset=["month_ts"])
    out["month_label"] = out["month_ts"].dt.strftime("%Y-%m")
    return out


def load_combined(
    xlsx_path: str | os.PathLike[str] | None = None,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load Combined Data sheet (availability % + runtime hours) with optional Parquet cache."""
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_COMBINED_XLSX
    if not path.is_file():
        raise FileNotFoundError(f"Excel not found: {path}")

    if use_cache and not force_refresh and COMBINED_CACHE_PATH.is_file():
        if COMBINED_CACHE_PATH.stat().st_mtime >= path.stat().st_mtime:
            df = pd.read_parquet(COMBINED_CACHE_PATH)
            _validate_combined_normalized(df)
            return df

    df = pd.read_excel(
        path,
        sheet_name=COMBINED_SHEET_NAME,
        engine="openpyxl",
        skiprows=[0],
        header=0,
        usecols=COMBINED_READ_COLUMNS,
    )
    _validate_combined_schema(df)
    df = _normalize_combined_frame(df)
    _validate_combined_normalized(df)

    if use_cache:
        COMBINED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(COMBINED_CACHE_PATH, index=False)

    return df
