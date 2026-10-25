"""Deterministically generate clearly labelled fake data for offline demonstrations only."""
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
            # Generated seasonal pattern is deliberately artificial; it is not derived from observations.
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
            # Synthetic proxy is only for code-path demonstration, not the CPCB AQI algorithm.
            proxy = max(pm25 * 1.1, pm10 * .55, no2 * 2.0, so2 * 2.0, co * 20, o3 * .8)
            aqi = float(np.clip(.65 * proxy + .35 * previous + rng.normal(0, 10), 8, 490))
            if rng.random() < .003:
                aqi = float(np.clip(aqi + rng.uniform(80, 150), 0, 500))
            previous = aqi
            vals = {"Date": d.date().isoformat(), "City": city, "PM2.5": round(pm25, 2), "PM10": round(pm10, 2),
                    "NO": round(no, 2), "NO2": round(no2, 2), "NOx": round(nox, 2), "NH3": round(nh3, 2),
                    "CO": round(co, 2), "SO2": round(so2, 2), "O3": round(o3, 2), "AQI": round(aqi, 1),
                    "AQI_Bucket": _bucket(aqi), "DataStatus": "SYNTHETIC_DEMO_NOT_OBSERVATIONS"}
            for key in ["PM2.5", "PM10", "NO2", "SO2", "O3"]:
                if rng.random() < .035: vals[key] = np.nan
            rows.append(vals)
    df = pd.DataFrame(rows)
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
    return df


def _bucket(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def create_demo_titanic(path: str | Path, seed: int = 3, n: int = 500):
    rng=np.random.default_rng(seed); sex=rng.choice(["female","male"],n,p=[.49,.51]); pclass=rng.choice([1,2,3],n,p=[.24,.21,.55]); age=np.clip(rng.normal(30,14,n),0.4,78); sib=rng.poisson(.5,n); parch=rng.poisson(.35,n)
    logits=1.3*(sex=="female")+.7*(pclass==1)+.15*(pclass==2)-.025*(age-25)-.45*(sib>2)
    prob=1/(1+np.exp(-logits)); survived=rng.binomial(1,prob)
    df=pd.DataFrame({"PassengerId":np.arange(1,n+1),"Survived":survived,"Pclass":pclass,"Sex":sex,"Age":age.round(1),"SibSp":sib,"Parch":parch,"Fare":np.round(np.maximum(4,rng.lognormal(3.0-(pclass-1)*.5,0.65,n)),2),"Embarked":rng.choice(["S","C","Q"],n,p=[.72,.19,.09]),"DataStatus":"SYNTHETIC_DEMO_NOT_OBSERVATIONS"})
    df.loc[rng.random(n)<.08,"Age"]=np.nan; Path(path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); return df


def create_demo_students(path: str | Path, seed: int = 11, n: int = 500):
    rng=np.random.default_rng(seed); gender=rng.choice(["female","male"],n); parental=rng.choice(["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"],n,p=[.12,.2,.22,.2,.16,.1]); study=np.clip(rng.normal(2.5,1.2,n),0,8); prep=rng.choice(["none","completed"],n,p=[.64,.36]); lunch=rng.choice(["standard","free/reduced"],n,p=[.65,.35]); edu=pd.Series(parental).map({"some high school":-6,"high school":-3,"some college":0,"associate's degree":2,"bachelor's degree":5,"master's degree":7}).to_numpy(); prep_bonus=(prep=="completed")*5; ability=rng.normal(0,10,n)
    math=np.clip(62+edu+study*1.8+prep_bonus+ability+rng.normal(0,9,n),0,100); reading=np.clip(67+edu*.8+study*1.2+prep_bonus+ability*.7+rng.normal(0,8,n),0,100); writing=np.clip(65+edu*.9+study*1.3+prep_bonus+ability*.65+rng.normal(0,9,n),0,100)
    df=pd.DataFrame({"gender":gender,"race/ethnicity":rng.choice(["group A","group B","group C","group D","group E"],n,p=[.08,.19,.32,.28,.13]),"parental level of education":parental,"lunch":lunch,"test preparation course":prep,"study hours":study.round(1),"math score":math.round().astype(int),"reading score":reading.round().astype(int),"writing score":writing.round().astype(int),"DataStatus":"SYNTHETIC_DEMO_NOT_OBSERVATIONS"})
    Path(path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); return df


def create_demo_boston(path: str | Path, seed: int = 20, n: int = 506):
    rng=np.random.default_rng(seed); rm=rng.normal(6.2,.7,n).clip(3.5,8.5); lstat=rng.gamma(3.5,2.2,n).clip(1,38); ptratio=rng.normal(18.5,2,n).clip(12,22); nox=rng.normal(.55,.12,n).clip(.35,.9); crim=rng.lognormal(-1.4,1.1,n).clip(.005,80); tax=rng.normal(400,150,n).clip(180,720); dis=rng.normal(3.8,2,n).clip(1,12)
    medv=np.clip(24+5.2*(rm-6)-.65*lstat-.7*(ptratio-18)-10*(nox-.55)-.025*(tax-400)+rng.normal(0,3,n),5,50)
    df=pd.DataFrame({"CRIM":crim,"ZN":rng.choice([0,12,25,40],n),"INDUS":rng.uniform(1,28,n),"CHAS":rng.binomial(1,.07,n),"NOX":nox,"RM":rm,"AGE":rng.uniform(10,100,n),"DIS":dis,"RAD":rng.integers(1,25,n),"TAX":tax,"PTRATIO":ptratio,"B":rng.uniform(0,400,n),"LSTAT":lstat,"MEDV":medv,"DataStatus":"SYNTHETIC_DEMO_NOT_OBSERVATIONS"})
    Path(path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); return df
