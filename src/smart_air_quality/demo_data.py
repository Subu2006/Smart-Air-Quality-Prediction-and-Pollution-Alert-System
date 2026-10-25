"""Deterministically generate clearly labelled fallback synthetic air quality data for offline demonstrations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def create_demo_air_quality(path: str | Path, seed: int = 273) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = {"Delhi": 90, "Lucknow": 67, "Jaipur": 55, "Amritsar": 52, "Mumbai": 38, "Bengaluru": 28}
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    rows = []
    for city, base in cities.items():
        previous = float(base + 30)
        for d in dates:
            winter = 1.0 if d.month in (11, 12, 1, 2) else 0.0
            monsoon = 1.0 if d.month in (6, 7, 8, 9) else 0.0
            weekly = 4 * np.sin(2 * np.pi * d.dayofweek / 7)
            seasonal = 18 * winter - 10 * monsoon + 5 * np.sin(2 * np.pi * d.dayofyear / 365.25)
            pm25 = max(4, base * 0.65 + seasonal + weekly + rng.normal(0, 12))
            pm10 = max(8, pm25 * (1.55 + rng.normal(0, .12)) + rng.normal(0, 8))
            no2 = max(2, base * .18 + 6 * winter + rng.normal(0, 5))
            no = max(1, no2 * .5 + rng.normal(0, 3))
            nox = max(2, no2 * 1.25 + rng.normal(0, 4))
            nh3 = max(1, base * .08 + rng.normal(0, 3))
            co = max(.1, base * .015 + rng.normal(0, .5))
            so2 = max(1, base * .1 + rng.normal(0, 3))
            o3 = max(2, 20 + 15 * (1-monsoon) + rng.normal(0, 8))
            proxy = max(pm25 * 1.1, pm10 * .55, no2 * 2.0, so2 * 2.0, co * 20, o3 * .8)
            aqi = float(np.clip(.65 * proxy + .35 * previous + rng.normal(0, 10), 8, 490))
            if rng.random() < .003:
                aqi = float(np.clip(aqi + rng.uniform(80, 150), 0, 500))
            previous = aqi
            vals = {
                "Date": d.date().isoformat(), "City": city, "PM2.5": round(pm25, 2), "PM10": round(pm10, 2),
                "NO": round(no, 2), "NO2": round(no2, 2), "NOx": round(nox, 2), "NH3": round(nh3, 2),
                "CO": round(co, 2), "SO2": round(so2, 2), "O3": round(o3, 2), "AQI": round(aqi, 1),
                "AQI_Bucket": _bucket(aqi), "DataStatus": "SYNTHETIC_DEMO_NOT_OBSERVATIONS"
            }
            for key in ["PM2.5", "PM10", "NO2", "SO2", "O3"]:
                if rng.random() < .035:
                    vals[key] = np.nan
            rows.append(vals)
    df = pd.DataFrame(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def _bucket(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"
