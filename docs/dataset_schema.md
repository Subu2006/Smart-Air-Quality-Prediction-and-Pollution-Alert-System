# Data schema and provenance

## Primary training input: city-day CSV

Required columns:

| Column | Type | Purpose | Handling |
|---|---|---|---|
| `Date` | Date-like | Observation date | Invalid dates are counted and dropped from modeling. |
| `City` | Text | Location/group key | Trim whitespace; empty values are invalid. |
| `AQI` | Numeric | Reported source target/index | Missing remains missing; values outside 0–500 are flagged, never silently clipped. |
| `AQI_Bucket` | Text, optional | Supplied category label | Retained as descriptive information; numeric AQI bands are determined using a separate CPCB reference. |
| `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene` | Numeric, optional | Pollutant concentrations | Converted to numbers; blanks stay missing. Verify units and averaging period in the original source. |
| `DataStatus` | Text, optional | Demo provenance marker | `SYNTHETIC_DEMO...` marks generated rows; never interpret these values as measurements. |

The bundled generated CSV has 8,766 rows, six generated city labels and dates from 2021-01-01 through 2024-12-31. Those are synthetic design choices, not observed coverage. Synthetic AQI values are produced using an artificial proxy and persistence/noise process; they do **not** implement the official CPCB pollutant sub-index calculation.

## Public candidate source

Rohan Rao's India air-quality dataset is distributed through Kaggle. TUMI's data portal lists an India urban air-pollution dataset based on CPCB data and lists CC0 1.0 licensing metadata. The exact file version and field availability must be checked at download time. The original supplied reference report used city-day and station-day data and highlighted the need to distinguish their grains. This package's initial model is intentionally limited to city-day forecasting.

## Validation rules

- Required columns present.
- Parse date, trim location string and numerically parse AQI/pollutant columns.
- Report `Date` min/max, city count, duplicates, missingness and AQI out-of-range rows.
- City-date duplicate key is deduplicated to the first row by the loader. Review duplicates in source audits; do not apply this behavior to multiple station observations without changing the key.
- No blanket deletion of outliers, no blanket replacement of missing measurements with zero, and no claim that source AQI is current.
- Next-day training label is `AQI[t+1]` only when the following calendar date exists for the same city. Predictors use measurements observed on date t; pollutants from t+1 are not used.
