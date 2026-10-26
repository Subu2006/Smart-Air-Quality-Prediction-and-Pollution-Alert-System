# 📊 CPCB Ambient Air Quality Monitoring Dataset Schema

## 1. Overview
This dataset contains continuous ambient air quality monitoring telemetry records collected by the **Central Pollution Control Board (CPCB)** of India across 26 major metropolitan and industrial urban centers between **1 January 2015 and 1 July 2020** (29,531 total daily records).

---

## 2. Feature Data Dictionary

| Column Name | Physical Data Type | Measurement Unit | CPCB NAAQS Standard (24-hr) | WHO Guideline (2021) | Environmental & Health Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **City** | Categorical String | N/A | N/A | N/A | Target metropolitan monitoring region (26 cities). |
| **Date** | Datetime (ISO 8601) | YYYY-MM-DD | N/A | N/A | Daily observation timestamp. |
| **PM2.5** | Continuous Float | $\mu\text{g/m}^3$ | $60\ \mu\text{g/m}^3$ | $15\ \mu\text{g/m}^3$ | Fine inhalable particles ($< 2.5\ \mu\text{m}$); primary driver of Indian AQI. Translocates into vascular circulation. |
| **PM10** | Continuous Float | $\mu\text{g/m}^3$ | $100\ \mu\text{g/m}^3$ | $45\ \mu\text{g/m}^3$ | Coarse respirable dust and mechanical abrasion particles ($< 10\ \mu\text{m}$). Spikes in pre-monsoon spring dust storms. |
| **NO** | Continuous Float | $\mu\text{g/m}^3$ | Monitored | Monitored | Nitric oxide; primary vehicle exhaust byproduct rapidly oxidized to $\text{NO}_2$. |
| **NO2** | Continuous Float | $\mu\text{g/m}^3$ | $80\ \mu\text{g/m}^3$ | $25\ \mu\text{g/m}^3$ | Nitrogen dioxide; potent pulmonary irritant and photochemical ozone/secondary aerosol precursor. |
| **NOx** | Continuous Float | $\text{ppb}$ | Monitored | Monitored | Total nitrogen oxide oxides; transit corridor combustion indicator. |
| **NH3** | Continuous Float | $\mu\text{g/m}^3$ | $400\ \mu\text{g/m}^3$ | Monitored | Ammonia; agricultural fertilizer emissions and domestic waste precursor to ammonium nitrate aerosol. |
| **CO** | Continuous Float | $\text{mg/m}^3$ | $2.0\ \text{mg/m}^3$ (8-hr) | $4.0\ \text{mg/m}^3$ | Carbon monoxide; incomplete carbon combustion asphyxiant that binds blood hemoglobin. |
| **SO2** | Continuous Float | $\mu\text{g/m}^3$ | $80\ \mu\text{g/m}^3$ | $40\ \mu\text{g/m}^3$ | Sulphur dioxide; emitted from coal-fired power stations and heavy industrial fuel oil. |
| **O3** | Continuous Float | $\mu\text{g/m}^3$ | $100\ \mu\text{g/m}^3$ (8-hr) | $100\ \mu\text{g/m}^3$ | Ground-level tropospheric ozone; secondary photochemical oxidant formed via $\text{NO}_x + \text{VOCs} + \text{UV light}$. |
| **Benzene** | Continuous Float | $\mu\text{g/m}^3$ | $5.0\ \mu\text{g/m}^3$ (Annual) | Non-threshold | Carcinogenic volatile organic compound from petroleum refineries and vehicular evaporation. |
| **Toluene** | Continuous Float | $\mu\text{g/m}^3$ | Monitored | Monitored | Industrial solvent VOC and photochemical precursor. |
| **Xylene** | Continuous Float | $\mu\text{g/m}^3$ | Monitored | Monitored | Aromatic hydrocarbon solvent; tracked along petrochemical industrial belts. |
| **AQI** | Continuous Float | Dimensionless Index | $\le 100$ (Acceptable) | Varies | Composite Central Pollution Control Board National Air Quality Index (0–500 scale). |
| **AQI_Bucket** | Categorical String | CPCB Severity Tier | Good ($\le 50$) | Varies | Official CPCB health tier: Good, Satisfactory, Moderate, Poor, Very Poor, Severe. |

---

## 3. Data Quality & Preprocessing Protocol
1. **Preservation of Legitimate Missingness:** Missing telemetry (ranging from 7.0% for CO to 61.3% for Xylene) is preserved during EDA and imputed via median values with binary missingness indicators during training. Missing values are **never** filled with zero.
2. **Sensor Saturation & Scale Exceedances ($AQI > 500$):** Under CPCB standard formulas, the nominal index cap is 500. However, 543 historical records exceed 500 (reaching 2,049 in Delhi/Ahmedabad). These records are tagged with an `Out of range` data quality audit flag while retaining genuine disaster values for training.
3. **Chronological Pair Construction:** Target $y_{c,t} = \text{AQI}_{c,t+1}$ is constructed strictly on consecutive calendar days ($t+1$). Discontinuous date gaps are excluded, preventing temporal interpolation artifacts.
