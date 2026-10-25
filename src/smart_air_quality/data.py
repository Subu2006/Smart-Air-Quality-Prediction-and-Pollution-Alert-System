"""Data loading, schema normalization and quality reporting for air-quality CSV files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
REQUIRED_COLUMNS = ["Date", "City", "AQI"]

@dataclass
class DataLoadResult:
    frame: pd.DataFrame
    audit: dict[str, Any]


def normalize_column_names(columns: list[str]) -> dict[str, str]:
    """Map column names case-insensitively to known canonical names without guessing unknown fields."""
    canonical = {c.lower().replace(" ", "").replace("_", ""): c for c in ["Date", "City", "AQI", "AQI_Bucket", *POLLUTANTS]}
    out = {}
    for col in columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")
        out[col] = canonical.get(key, str(col).strip())
    return out


def load_air_quality_csv(path: str | Path, *, deduplicate: bool = True) -> DataLoadResult:
    """Read and normalize a source CSV; preserve missing measurements rather than replacing them with zero."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See docs/dataset_schema.md for acquisition instructions.")
    frame = pd.read_csv(path)
    original_rows = len(frame)
    original_columns = list(frame.columns)
    frame = frame.rename(columns=normalize_column_names(original_columns))
    missing_required = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_required:
        raise ValueError(f"Required columns missing: {missing_required}. Required: {REQUIRED_COLUMNS}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["City"] = frame["City"].astype("string").str.strip()
    for col in ["AQI", *[p for p in POLLUTANTS if p in frame.columns]]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    invalid_dates = int(frame["Date"].isna().sum())
    blank_cities = int(frame["City"].isna().sum() + frame["City"].eq("").sum())
    duplicate_mask = frame.duplicated(subset=["City", "Date"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if deduplicate and duplicate_count:
        frame = frame.loc[~duplicate_mask].copy()
    frame = frame.loc[frame["Date"].notna() & frame["City"].notna() & frame["City"].ne("")].copy()
    frame = frame.sort_values(["City", "Date"]).reset_index(drop=True)
    if "AQI_Bucket" in frame.columns:
        frame["AQI_Bucket"] = frame["AQI_Bucket"].astype("string").str.strip()
    demo_flag = False
    if "DataStatus" in frame.columns:
        demo_flag = bool(frame["DataStatus"].astype(str).str.contains("SYNTHETIC_DEMO", case=False, na=False).any())
    aqi = frame["AQI"]
    audit = {
        "file": path.name,
        "data_kind": "synthetic_demo" if demo_flag else "external_or_user_supplied",
        "rows_input": int(original_rows),
        "rows_after_validation": int(len(frame)),
        "columns_input": original_columns,
        "missing_required_columns": missing_required,
        "invalid_date_rows": invalid_dates,
        "blank_city_rows": blank_cities,
        "duplicate_city_date_rows": duplicate_count,
        "date_min": str(frame["Date"].min().date()) if len(frame) else None,
        "date_max": str(frame["Date"].max().date()) if len(frame) else None,
        "cities": int(frame["City"].nunique()),
        "missing_aqi": int(aqi.isna().sum()),
        "aqi_out_of_range_rows": int(((aqi < 0) | (aqi > 500)).sum()),
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "notes": ["Missing measurements remain missing; they are not treated as zero.",
                  "Deduplication uses City + Date and keeps the first occurrence; review duplicates before production use.",
                  "AQI range flags use the CPCB 0-500 reporting scale; unusual rows are flagged, not silently clipped."],
    }
    return DataLoadResult(frame=frame, audit=audit)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
