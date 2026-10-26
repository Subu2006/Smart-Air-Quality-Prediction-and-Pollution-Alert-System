# 🛡️ Model Validation & Experimental Protocol Summary

## 1. Zero Temporal Data Leakage Strategy
Time-series atmospheric phenomena cannot be evaluated via random $k$-fold cross-validation, which causes massive lookahead data contamination. This project enforces strict **chronological train-test splitting**:

- **Outer Cutoff Date:** `2019-05-27`
- **Training Window:** `2015-01-01` to `2019-05-26` (15,655 records; 64.3%)
- **Held-Out Test Window:** `2019-05-27` to `2020-07-01` (8,688 records; 35.7%)
- **Internal Validation Window:** Chronological last 20% of the training dates (`2018-07-16` to `2019-05-26`) used strictly for hyperparameter selection and candidate model ranking prior to outer test evaluation.

---

## 2. Held-Out Test Evaluation Matrix (Outer Split: 8,688 Records)

| Model Architecture | MAE (AQI) | RMSE (AQI) | $R^2$ Score | Exact Category Accuracy | Macro F1 | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **19.66** | **40.04** | **0.8508** | **75.79%** | **0.6387** | **Selected Final Model** |
| **Gradient Boosting (GBDT)** | 20.29 | 41.69 | 0.8382 | 73.98% | 0.5973 | Benchmark Candidate |
| **Persistence Baseline ($AQI_t$)** | 22.10 | 47.27 | 0.7920 | 74.17% | 0.6248 | Serial Autocorrelation Benchmark |
| **Decision Tree Regressor** | 22.36 | 48.94 | 0.7771 | 73.61% | 0.6311 | Non-parametric Baseline |
| **Ridge Regression ($L_2$)** | 24.92 | 41.38 | 0.8406 | 65.93% | 0.5897 | Regularized Linear Model |
| **Linear Regression (OLS)** | 25.06 | 41.49 | 0.8397 | 65.83% | 0.5892 | Unregularized Linear Model |

---

## 3. Empirical Residual Uncertainty Bounds
To prevent misleading single-point certainty, empirical forecast intervals are constructed from held-out test residuals:
$$\sigma_{\text{res}} = \sqrt{\frac{1}{N_{test}} \sum_{i=1}^{N_{test}} (y_i - \hat{y}_i)^2} = 39.63\ \text{AQI points}$$
$$95\%\ \text{Confidence Interval} = [\hat{y} \pm 1.96 \cdot \sigma_{\text{res}}] = [\hat{y} \pm 77.68\ \text{AQI points}]$$

---

## 4. Multi-Tier Alert Engine Specifications
- **Data Quality Saturation Alert:** Triggered when recorded or predicted $\text{AQI} > 500$.
- **Statutory Threshold Exceedance:** Triggered when recorded observation exceeds user-configured thresholds ($\ge 200$ for Poor, $\ge 300$ for Very Poor, $\ge 400$ for Severe).
- **24-Hour Predictive Early Warning:** Dispatched 24 hours in advance to enable proactive civic interventions (school recess cancellations, dust misting cannon mobilization, and hospital respiratory ward readiness).
