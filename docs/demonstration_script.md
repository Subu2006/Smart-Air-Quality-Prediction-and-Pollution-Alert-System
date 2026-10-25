# Faculty demonstration script (approximately 5 minutes)

1. **Trust first (30 seconds).** Show the banner and state that the included dataset/model are generated demo artifacts; no official data CSV was attached to the build. Explain how to point the pipeline at verified real data.
2. **Explore (45 seconds).** Select a city and date range, compare AQI and PM2.5/PM10 time series, then export filtered rows.
3. **Explain the data (45 seconds).** Show a histogram, city comparison and correlation heatmap. Say that correlation does not establish causation and that synthetic patterns are only code-path demonstrations.
4. **Predict (60 seconds).** Select the latest record for a city; run one-day-ahead estimate. Trace predictors at time t and target AQI at time t+1. Note that the shipped model is demo-trained.
5. **Alert behavior (45 seconds).** Change the threshold; distinguish `RECORDED_THRESHOLD`, `PREDICTED_THRESHOLD`, `DATA_QUALITY` and no-alert conditions. Point out that no push messages are sent.
6. **Model evidence (45 seconds).** Discuss the chronological test, validation-based model selection and persistence benchmark; all demo metrics are synthetic.
7. **Syllabus proof (45 seconds).** Open the coverage matrix and scripts. Mention that Titanic/student/Boston practical files are separate supplementary exercises rather than modules in the air-quality production dashboard.
