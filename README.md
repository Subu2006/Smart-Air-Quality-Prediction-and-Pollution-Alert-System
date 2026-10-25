# 🌍 SMART AIR QUALITY PREDICTION AND POLLUTION ALERT SYSTEM

### An Enterprise-Grade, Machine Learning-Driven Atmospheric Early Warning Platform
**Course:** CSE273: Foundations of Machine Learning (Session 2026–2027)  
**Academic Institution:** Lovely School of Computer Science & Engineering, Lovely Professional University  
**Student Author:** Subrat Panigrahi | **Registration Number:** 26143073 | **Section:** N1  
**Faculty Guidance:** Course Faculty / Evaluator  
**GitHub Repository:** [Subu2006/Smart-Air-Quality-Prediction-and-Pollution-Alert-System](https://github.com/Subu2006/Smart-Air-Quality-Prediction-and-Pollution-Alert-System)  
**Reference Benchmark:** [AIRWATCH INDIA](https://github.com/Subu2006/Airwatch-India-PowerBi)

---

## 🌟 Executive Summary & Vision

The **Smart Air Quality Prediction and Pollution Alert System** is an end-to-end, scientifically defensible Machine Learning software platform engineered to forecast ambient air quality and dispatch automated public health early warnings across 26 major Indian metropolitan centers. 

Moving fundamentally beyond descriptive business intelligence dashboards (which merely visualize past pollution after citizen exposure has already occurred), this system delivers **24-hour predictive foresight**:
- Ingests **29,531 daily continuous observation records** (2015–2020) across criteria particulate matter ($\text{PM}_{2.5}$, $\text{PM}_{10}$) and gaseous pollutants ($\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{NH}_3$).
- Enforces strict **chronological train-test validation** (15,655 train vs. 8,688 out-of-sample test records) to guarantee **zero temporal data leakage**.
- Benchmarks six candidate algorithm paradigms: **Random Forest Regressor** achieves state-of-the-art predictive performance ($R^2 = 0.8508$, $\text{MAE} = 19.66$ AQI points, $\text{RMSE} = 40.04$, and **75.79%** exact CPCB category accuracy), significantly outperforming the Persistence Baseline ($R^2 = 0.7920$).
- Quantifies forecast uncertainty via **95% empirical prediction intervals** ($\pm 77.68$ AQI points).
- Embeds core mathematical foundations directly into atmospheric analysis: **PCA eigenvalue decomposition**, **Welch's two-sample $t$-test hypothesis testing**, and **Gradient Descent optimization dynamics** (SGD, Polyak Momentum, RMSprop, Adam).
- Features an automated, multi-tier **CPCB-compliant Pollution Alert Engine** with simulated multi-channel public dispatch (SMS, Webhooks, and Municipal Emergency Notices).
- Includes an interactive **Streamlit production dashboard** and an automated compiler generating the faculty-mandated **30-page academic project report** in Microsoft Word (`.docx`) and Adobe PDF (`.pdf`).

---

## 🚀 Key Architectural Capabilities

1. **Automated Data Engineering & Quality Auditing:**
   - Ingests multi-year CPCB monitoring records with strict schema validation and date parsing.
   - Preserves legitimate telemetry missingness without artificial zero-imputation; rigorously audits sensor saturation ($AQI > 500$, 543 historical occurrences) with dedicated quality flags.
2. **Leakage-Free Temporal Feature Engineering:**
   - Cyclical harmonic transformations ($\sin, \cos$ day-of-year) capturing non-linear seasonal cycles.
   - True single-day lag pairing guaranteeing complete temporal separation between observation day $t$ and target day $t+1$.
3. **Core Mathematical Foundations Lab:**
   - **Linear Algebra (Unit I & IV):** Covariance matrix $\Sigma = \frac{1}{N-1} X^T X$ and eigenvalue decomposition for criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3); Scree plot and loadings biplot.
   - **Probability & Statistics (Unit IV & V):** Welch's two-sample $t$-test and Mann-Whitney $U$ test evaluating Winter vs Monsoon pollution disparities ($p < 10^{-15}$); 95% confidence intervals.
   - **Optimization Methods (Unit V):** Multi-optimizer convergence simulator (Batch GD, SGD, Momentum, RMSprop, Adam) minimizing Mean Squared Error on the atmospheric loss surface.
4. **Out-of-Sample Machine Learning Suite:**
   - Evaluates Persistence Baseline, Linear Regression (OLS), $L_2$ Regularized Ridge Regression, Decision Trees, Random Forests, and Gradient Boosting.
   - Hyperparameter selection conducted on internal chronological validation window before outer test evaluation.
5. **Empirical Residual Uncertainty Bounds:**
   - Derives 95% confidence bounds ($[\hat{y} \pm 1.96\sigma]$) from held-out residual error variance, avoiding deceptive single-point certainty.
6. **Automated Multi-Tier Public Alert Engine:**
   - Generates Data Quality Flags, Statutory Exceedances, and 24-Hour Predictive Public Warnings mapped to official CPCB health impact categories with tailored demographic advisories.
7. **Interactive Production Web Application:**
   - Modern, high-contrast Streamlit interface featuring 8 specialized operational modules, geospatial station mapping, live inference, and alert payload simulation.
8. **Publication-Grade Academic Report Compiler:**
   - Compiles the complete 30-page academic project report matching the faculty whiteboard allocation, formatted with custom tables, callout boxes, and 10 real data figures.

---

## 🛠️ Technology Stack

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.10+ (Tested on Python 3.14.7) |
| **Data Engineering** | NumPy 2.5+, Pandas 3.0+ |
| **Machine Learning** | Scikit-Learn 1.9+, SciPy 1.18+, Joblib 1.6+ |
| **Mathematical Lab** | Linear Algebra (`numpy.linalg`), Statistical Inference (`scipy.stats`) |
| **Data Visualization** | Matplotlib 3.11+, Seaborn 0.13+ |
| **Web Application** | Streamlit 1.63+ |
| **Quality Assurance** | Pytest 9.1+ |
| **Document Compilation** | Python-Docx 1.2+, Docx2pdf 0.1.8, PyWin32 312 |

---

## 📁 Repository Directory Structure

```text
Smart Air Quality Prediction and Pollution Alert System/
├── app.py                          # Streamlit Interactive Web Application (8 modules)
├── train_model.py                  # Reproducible model training & chronological evaluation script
├── run_analysis.py                 # Full exploratory analysis, PCA, and figure generation script
├── build_report.py                 # Academic Word (.docx) & PDF (30-page) report compiler
├── requirements.txt                # Python library dependency specifications
├── README.md                       # Comprehensive system documentation
│
├── data/
│   ├── raw/                        # Authentic CPCB monitoring datasets
│   │   ├── city_day.csv            # 29,531 daily city observation records (2015-2020)
│   │   ├── station_day.csv         # 108,035 station-level daily records
│   │   ├── stations.csv            # 230 CPCB monitoring station catalog
│   │   └── StationCoordinates.csv  # 230 station geographic coordinates for mapping
│   └── demo/                       # Deterministic offline fallback datasets
│
├── src/smart_air_quality/          # Modular Python source package
│   ├── __init__.py                 # Clean package API exports
│   ├── data.py                     # Schema validation, missingness preservation & audit
│   ├── features.py                 # Calendar harmonics & chronological feature builders
│   ├── math_foundations.py         # PCA, eigenvalue decomposition, t-tests & optimizers
│   ├── models.py                   # Model pipelines, chronological benchmarking & inference
│   ├── alerts.py                   # CPCB AQI categories & threshold rule evaluation
│   └── openaq.py                   # Live OpenAQ API client with graceful fallback
│
├── models/                         # Serialized ML artifacts & evaluation logs
│   ├── air_quality_model.joblib    # Fitted Random Forest Regressor artifact
│   ├── air_quality_model.json      # Metadata, audit ledger, and evaluation metrics
│   ├── air_quality_model.csv       # Test set candidate model benchmark comparison table
│   └── air_quality_model_validation.csv # Validation set model selection table
│
├── notebooks/                      # End-to-end interactive Jupyter notebooks
│   └── smart_air_quality_pipeline.ipynb # Master reproducible pipeline walkthrough
│
├── reports/                        # Publication figures & compiled reports
│   ├── figures/                    # 10 publication-quality analytical figures (PNG)
│   ├── results_summary.json        # Structured statistical and modeling summary
│   ├── Smart_Air_Quality_Project_Report.docx # Formatted Word report (30 pages)
│   └── Smart_Air_Quality_Project_Report.pdf  # Compiled Adobe PDF report (30 pages)
│
├── tests/                          # Automated unit test suite
│   └── test_pipeline.py            # Comprehensive tests (data, models, alerts, math)
└── docs/                           # Architecture and schema documentation
    ├── dataset_schema.md           # Field dictionaries and physical units
    └── validation_summary.md       # Validation strategy and data quality rules
```

---

## ⚡ Quickstart & Execution Guide

### 1. Environment Setup
Clone the repository and initialize a clean Python virtual environment:
```bash
git clone https://github.com/Subu2006/Smart-Air-Quality-Prediction-and-Pollution-Alert-System.git
cd "Smart Air Quality Prediction and Pollution Alert System"

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train Models with Chronological Validation
Train all six candidate algorithms, evaluate out-of-sample test performance, and serialize artifacts:
```bash
python train_model.py
```
*Output Summary:*
```text
Candidate Model Performance (Chronological Outer Test Split: N = 8,688):
Model                          MAE      RMSE     R2       Cat Acc    Macro F1  
----------------------------------------------------------------------------
Random Forest (Bagging)        19.66    40.04    0.8508   0.7579     0.6387    
Gradient Boosting (GBDT)       20.29    41.69    0.8382   0.7398     0.5973    
Persistence baseline (AQI_t)   22.10    47.27    0.7920   0.7417     0.6248    
Decision Tree Regressor        22.36    48.94    0.7771   0.7361     0.6311    
Ridge Regression (L2)          24.92    41.38    0.8406   0.6593     0.5897    
Linear Regression (OLS)        25.06    41.49    0.8397   0.6583     0.5892    
```

### 3. Generate Analytical Figures & Mathematical Artifacts
Run exploratory analysis, PCA eigenvalue decomposition, and optimizer loss simulations:
```bash
python run_analysis.py
```
*Generated Figures in `reports/figures/`:*
- `fig01_national_aqi_distribution.png`: Histogram, boxplot, and CPCB bands (Skewness 3.40, Kurtosis 21.42).
- `fig02_pollutant_correlation_matrix.png`: Criteria pollutant Pearson correlation matrix ($r=0.88$ for PM2.5).
- `fig03_temporal_monthly_seasonal.png`: Monthly lines across years and seasonal bar chart (Winter vs Monsoon).
- `fig04_city_pollution_rankings.png`: Top 10 most polluted vs cleanest monitored cities.
- `fig05_model_performance_comparison.png`: MAE, RMSE, and $R^2$ benchmark across all 6 models.
- `fig06_predicted_vs_actual_residuals.png`: Actual vs predicted parity plot and zero-mean residual distribution.
- `fig07_feature_importance.png`: Relative feature importance ranked by Gini impurity reduction.
- `fig08_cpcb_category_distribution.png`: National monitoring days breakdown across CPCB categories.
- `fig09_pca_scree_and_biplot.png`: Scree plot of explained variance and PC1/PC2 pollutant factor loadings.
- `fig10_optimizer_convergence_curves.png`: Loss convergence of SGD, Momentum, RMSprop, and Adam on AQI loss.

### 4. Execute Automated Unit Test Suite
Verify data normalization, chronological splitting, alert rules, and mathematical functions:
```bash
python -m pytest tests/
```
*(All 7 unit tests pass with 100% success).*

### 5. Launch Interactive Streamlit Web Application
Run the multi-page production analytics and prediction application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 6. Compile Academic Project Report (Word & PDF)
Compile the comprehensive, 30-page academic report matching the faculty whiteboard allocation:
```bash
python build_report.py
```
Artifacts generated:
- `reports/Smart_Air_Quality_Project_Report.docx`
- `reports/Smart_Air_Quality_Project_Report.pdf`

---

## 📊 Academic Alignment: CSE273 Syllabus Integration

| Syllabus Unit | Mathematical & Algorithmic Principle | Implementation in Air Quality System | Status |
| :--- | :--- | :--- | :--- |
| **Unit I: ML Foundations & NumPy** | Supervised learning workflow, bias-variance tradeoff, NumPy array vectorization & broadcasting | Multi-pollutant feature vectors, data loading pipeline (`src/data.py`, `features.py`) | ✅ Implemented |
| **Unit II: Pandas & Matplotlib** | DataFrame manipulation, CSV ingestion, time-series aggregation, customized publication plotting | Multi-city exploratory analysis, time profiles, rolling trends (`run_analysis.py`) | ✅ Implemented |
| **Unit III: Exploratory Data Analysis** | Statistical moments (skewness, kurtosis), bivariate scatter/boxplots, Pearson correlation heatmaps | Distribution analysis (Skew: 3.40, Kurt: 21.42), inter-pollutant collinearity matrix | ✅ Implemented |
| **Unit IV: Linear Algebra & Calculus** | Covariance matrices, Eigenvalues/Eigenvectors, Principal Component Analysis (PCA), gradients | PCA decomposition of criteria pollutants, explained variance Scree plot (`math_foundations.py`) | ✅ Implemented |
| **Unit V: Statistics & Optimization** | Welch's t-test hypothesis testing, 95% Confidence Intervals, Gradient Descent (SGD, Momentum, RMSprop, Adam) | Seasonal hypothesis testing ($p < 10^{-15}$), interactive loss convergence simulator | ✅ Implemented |
| **Unit VI: Statistical Learning Theory** | Empirical Risk Minimization, non-parametric tree ensembles, bagging, boosting, chronological evaluation | Multi-model benchmark, Random Forest bagging, Gradient Boosting, residual uncertainty | ✅ Implemented |

---

## 🛡️ Responsible AI & Ethical Boundaries

1. **Non-Statutory Decision Support:** The predictions and alert flags generated by this system are analytical decision-support tools and do not substitute for official statutory forecasts issued by the Central Pollution Control Board (CPCB) or the Indian Meteorological Department (IMD).
2. **No Medical Diagnostic Claims:** Output air quality classifications indicate atmospheric criteria pollutant concentrations and should not be interpreted as individual medical diagnoses.
3. **Data Quality Transparency:** Extreme sensor saturation values ($AQI > 500$) and telemetry missingness are explicitly retained and audited rather than cosmetically smoothed.

---

## 📜 Academic Attribution & License
Developed by **Subrat Panigrahi** (Reg. No: 26143073, Section N1) for CSE273: Foundations of Machine Learning at Lovely Professional University. Distributed under the MIT License.
