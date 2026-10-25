"""Leakage-aware next-calendar-day AQI feature engineering."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import POLLUTANTS


def make_forecasting_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build one-day-ahead targets only where the next calendar day is observed for that city.

    Predictors include day-t AQI/pollutant values and calendar features known for t+1. No
    measurements from t+1 are used as predictors. Returns X, y and target-date index so that
    chronological splits are keyed to the day being predicted, not the preceding observation date.
    """
    if not {"Date", "City", "AQI"}.issubset(data.columns):
        raise ValueError("Forecasting requires Date, City and AQI columns")
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["Date", "City", "AQI"]).sort_values(["City", "Date"])
    grouped = df.groupby("City", sort=False)
    df["target_next_day"] = grouped["AQI"].shift(-1)
    df["target_date"] = grouped["Date"].shift(-1)
    df["target_gap_days"] = (df["target_date"] - df["Date"]).dt.days
    df["target_month"] = df["target_date"].dt.month
    df["target_dayofweek"] = df["target_date"].dt.dayofweek
    df["target_dayofyear_sin"] = np.sin(2 * np.pi * df["target_date"].dt.dayofyear / 365.25)
    df["target_dayofyear_cos"] = np.cos(2 * np.pi * df["target_date"].dt.dayofyear / 365.25)
    # Exactly next calendar day, not simply the next available row.
    df = df.loc[df["target_gap_days"].eq(1)].copy()
    numeric_features = ["AQI"] + [c for c in POLLUTANTS if c in df.columns]
    numeric_features += ["target_month", "target_dayofweek", "target_dayofyear_sin", "target_dayofyear_cos"]
    features = df[["City", *numeric_features]].copy()
    features["City"] = features["City"].astype(str)
    y = df["target_next_day"].astype(float)
    target_dates = df["target_date"].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.reset_index(drop=True), y.reset_index(drop=True), target_dates.reset_index(drop=True)


def split_chronologically(X: pd.DataFrame, y: pd.Series, target_dates: pd.Series, test_fraction: float = 0.2):
    """Chronological split by predicted target date; all locations for a date stay together."""
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    dates = pd.to_datetime(target_dates)
    unique_dates = np.sort(dates.dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("At least five distinct target dates are required for chronological evaluation")
    cut_index = min(max(int(len(unique_dates) * (1 - test_fraction)), 1), len(unique_dates) - 1)
    cutoff = pd.Timestamp(unique_dates[cut_index])
    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Chronological split produced an empty partition")
    return X.loc[train_mask].reset_index(drop=True), X.loc[test_mask].reset_index(drop=True), y.loc[train_mask].reset_index(drop=True), y.loc[test_mask].reset_index(drop=True), cutoff
