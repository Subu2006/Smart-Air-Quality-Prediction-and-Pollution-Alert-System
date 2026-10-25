"""Transparent Indian National AQI category and alert rules."""
from __future__ import annotations
import math
import pandas as pd

AQI_BANDS = [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
             (201, 300, "Poor"), (301, 400, "Very Poor"), (401, 500, "Severe")]


def aqi_category(value):
    if value is None or pd.isna(value):
        return "Unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(number) or number < 0 or number > 500:
        return "Out of range"
    # Forecast outputs are floating-point estimates, while the documented AQI bands
    # are usually printed with integer endpoints. Use inclusive upper bounds so a
    # value such as 50.4 enters Satisfactory rather than falling into an uncovered gap.
    if number <= 50:
        return "Good"
    if number <= 100:
        return "Satisfactory"
    if number <= 200:
        return "Moderate"
    if number <= 300:
        return "Poor"
    if number <= 400:
        return "Very Poor"
    return "Severe"


def evaluate_observation(city, timestamp, value, source="Recorded observation", threshold=200):
    """Create a data-quality alert or AQI threshold alert without presenting it as official warning."""
    cat = aqi_category(value)
    base = {"City": str(city), "Timestamp": str(timestamp), "Value": None if pd.isna(value) else float(value),
            "Category": cat, "SourceType": source, "Threshold": float(threshold)}
    if cat == "Unavailable":
        return {**base, "AlertType": "DATA_QUALITY", "Severity": "Information", "Reason": "AQI is missing or not numeric; no pollution classification generated."}
    if cat == "Out of range":
        return {**base, "AlertType": "DATA_QUALITY", "Severity": "High", "Reason": "AQI is outside the valid 0-500 reporting scale; verify the source record."}
    if float(value) >= threshold:
        sev = "High" if float(value) >= 301 else "Moderate"
        return {**base, "AlertType": "RECORDED_THRESHOLD", "Severity": sev,
                "Reason": f"Recorded AQI {float(value):.1f} meets or exceeds configured threshold {threshold:.1f}; category is {cat}."}
    return {**base, "AlertType": "NO_ALERT", "Severity": "Normal", "Reason": f"Recorded AQI is below configured threshold {threshold:.1f}."}


def build_alert_history(frame: pd.DataFrame, threshold: float = 200.0) -> pd.DataFrame:
    rows = [evaluate_observation(r.get("City", "Unknown"), r.get("Date", "Unknown"), r.get("AQI"), threshold=threshold)
            for _, r in frame.iterrows()]
    return pd.DataFrame(rows)


def evaluate_prediction(city, target_date, predicted_aqi, threshold=200.0, uncertainty=None):
    result = evaluate_observation(city, target_date, predicted_aqi, source="Model prediction (not an official warning)", threshold=threshold)
    if result["AlertType"] == "RECORDED_THRESHOLD":
        result["AlertType"] = "PREDICTED_THRESHOLD"
        result["Reason"] = result["Reason"].replace("Recorded AQI", "Predicted AQI") + " This is a model output, not an official warning."
    result["Uncertainty"] = uncertainty
    return result
