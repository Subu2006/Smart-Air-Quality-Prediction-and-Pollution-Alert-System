# Build and validation summary

**Build date:** 24 October 2026  
**Project:** Smart Air Quality Prediction and Pollution Alert System — CSE273

## Checks executed

| Check | Result | Evidence / boundary |
|---|---|---|
| Source syntax compilation | Passed | `python -m compileall -q app.py train_model.py run_analysis.py generate_demo_data.py build_report.py src practicals tests` |
| Automated unit/integration tests | Passed: 5 tests | `pytest -q`; covers loader/schema, missing-value preservation, chronological forecast framing, CPCB category boundaries/alerts, saved artifact loading and single prediction. |
| Practical 01–10 scripts | Passed | Each script was run from the project root. Outputs are saved under `reports/practicals/` where applicable. |
| Synthetic data audit | Passed for specified structural checks | 8,766 rows; 6 generated location labels; 8,766 retained rows; 8,760 next-day examples. This is synthetic test data only. |
| Train/evaluate/save/load | Passed | Ridge selected on validation MAE; model saved in `models/demo_model.joblib`, metadata and metrics stored beside it. |
| Forecast time split | Checked | Chronological partitions are grouped by the target date being predicted, not the preceding feature date. Final holdout begins on target date 2024-03-15. |
| Report rendering | Passed | Word report rendered to 40 page images and a PDF; full-page contact sheet reviewed for layout, page numbers, and major figures/tables. |
| Optional external API | Not run live | OpenAQ needs user credentials and network access; not represented as tested. |
| Browser/UI functional test | Not run | Streamlit was not installed in this build environment. Source syntax was checked; run with the declared requirements and manually follow `docs/demonstration_script.md`. |
| Real India-dataset experiment | Not run | The source CSV was not among supplied files and could not be downloaded in the build environment. No real-data metrics are claimed. |

## Demo-model metrics (synthetic only)

The forecastable frame contains 8,760 rows with a one-calendar-day-ahead target. The chronological outer split uses 7,008 training examples and 1,752 holdout examples. Ridge was selected on a separate validation period by minimum MAE.

| Model | MAE | RMSE | R² | Use |
|---|---:|---:|---:|---|
| Ridge | 9.829 | 13.477 | 0.562 | Selected on validation |
| Extra Trees | 9.979 | 13.612 | 0.553 | Candidate comparator |
| Random Forest | 10.205 | 13.837 | 0.538 | Candidate comparator |
| Persistence: AQI(t) | 11.820 | 16.196 | 0.367 | Baseline |

These scores are from generated toy data and are included only to show that model training, selection, metrics and artifact serialization operate. They must not be cited as evidence that the project predicts real air quality accurately.

## Outstanding before an authentic demonstration

1. Obtain a verified, licensed India air-quality CSV and preserve source/version metadata.
2. Run `python train_model.py --data data/raw/city_day.csv --model models/india_city_day_model.joblib --metadata models/india_city_day_model.json --data-kind real`.
3. Audit units, gaps, duplicate keys, outliers and source timestamps; then regenerate analysis and report evidence from the real file.
4. Install `requirements.txt`, launch the Streamlit UI, manually check every page/control and capture actual application screenshots if the faculty requires them.
5. Configure and test OpenAQ only when an API key, location ID, network access and current usage terms are available.
