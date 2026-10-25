from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smart_air_quality.data import load_air_quality_csv
from smart_air_quality.features import make_forecasting_frame, split_chronologically
from smart_air_quality.alerts import aqi_category, evaluate_observation, evaluate_prediction
from smart_air_quality.models import train_model, load_model, make_single_prediction


def test_data_loader_preserves_missing_and_flags_demo():
    result=load_air_quality_csv(ROOT/'data/demo/air_quality_demo.csv')
    assert result.audit['data_kind']=='synthetic_demo'
    assert len(result.frame)>1000
    assert result.frame['PM2.5'].isna().sum()>0
    assert result.audit['aqi_out_of_range_rows']==0


def test_missing_schema_fails(tmp_path):
    p=tmp_path/'bad.csv'; pd.DataFrame({'Date':['2024-01-01'],'City':['x']}).to_csv(p,index=False)
    with pytest.raises(ValueError): load_air_quality_csv(p)


def test_forecast_is_next_calendar_day_and_time_split():
    df=pd.DataFrame({'Date':pd.date_range('2020-01-01',periods=10).tolist()+pd.date_range('2020-01-01',periods=10).tolist(),
                     'City':['A']*10+['B']*10,'AQI':list(range(20,30))+list(range(40,50))})
    X,y,dates=make_forecasting_frame(df)
    assert len(X)==18
    assert X.columns.tolist()[0]=='City'
    Xtr,Xte,ytr,yte,cut=split_chronologically(X,y,dates)
    # `dates` are target dates (not observation dates), preventing the first held-out target
    # from also appearing among training labels. Every location for a target date stays together.
    date_series = pd.to_datetime(dates)
    assert date_series.loc[date_series < cut].max() < cut
    assert date_series.loc[date_series >= cut].min() == cut
    assert set(date_series[date_series < cut]).isdisjoint(set(date_series[date_series >= cut]))


def test_aqi_categories_and_bad_values():
    assert aqi_category(0)=='Good'; assert aqi_category(50)=='Good'; assert aqi_category(51)=='Satisfactory'
    assert aqi_category(100)=='Satisfactory'; assert aqi_category(101)=='Moderate'; assert aqi_category(500)=='Severe'
    assert aqi_category(50.4)=='Satisfactory'; assert aqi_category(100.4)=='Moderate'; assert aqi_category(200.4)=='Poor'
    assert aqi_category(501)=='Out of range'; assert aqi_category(None)=='Unavailable'
    assert evaluate_observation('A','2024-01-01',501)['AlertType']=='DATA_QUALITY'
    assert evaluate_observation('A','2024-01-01',230,threshold=200)['AlertType']=='RECORDED_THRESHOLD'
    assert evaluate_prediction('A','2024-01-02',320,threshold=200)['AlertType']=='PREDICTED_THRESHOLD'


def test_train_load_predict(tmp_path):
    model_path=tmp_path/'trained.joblib'; meta_path=tmp_path/'trained.json'
    meta=train_model(ROOT/'data/demo/air_quality_demo.csv',model_path,meta_path,data_kind='synthetic_demo',random_state=42)
    model=load_model(model_path)
    df=pd.read_csv(ROOT/'data/demo/air_quality_demo.csv')
    row=df.loc[df['City'].eq('Delhi')].sort_values('Date').iloc[-2]
    values={c:row.get(c) for c in ['AQI','PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']}
    pred=make_single_prediction(model,'Delhi',row['Date'],values,meta)
    assert np.isfinite(pred['predicted_aqi'])
    assert pred['prediction_date'] > pred['observation_date']
    assert meta['dataset_kind']=='synthetic_demo'
    assert (meta_path.with_suffix('.csv')).exists()


def test_real_model_artifact_and_inference():
    real_model_path = ROOT / "models/air_quality_model.joblib"
    real_meta_path = ROOT / "models/air_quality_model.json"
    if real_model_path.exists() and real_meta_path.exists():
        model = load_model(real_model_path)
        meta = json.loads(real_meta_path.read_text())
        assert meta["rows_used_for_forecasting"] > 20000
        assert meta["train_rows"] > 10000
        assert meta["test_rows"] > 5000
        assert meta["residual_std"] > 0
        
        # Test inference with uncertainty interval
        test_values = {"AQI": 150.0, "PM2.5": 65.0, "PM10": 110.0, "NO2": 28.0, "SO2": 14.0, "CO": 1.5, "O3": 35.0}
        pred = make_single_prediction(model, "Delhi", "2020-06-01", test_values, meta)
        assert np.isfinite(pred["predicted_aqi"])
        assert pred["lower_bound_95"] <= pred["predicted_aqi"] <= pred["upper_bound_95"]
        assert pred["category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def test_math_foundations_pca_and_hypothesis():
    from smart_air_quality.math_foundations import compute_air_quality_pca, perform_seasonal_hypothesis_test, simulate_optimizers_on_aqi_regression
    demo_path = ROOT / "data/demo/air_quality_demo.csv"
    if demo_path.exists():
        df = pd.read_csv(demo_path)
        pca_out = compute_air_quality_pca(df)
        assert "eigenvalues" in pca_out
        assert len(pca_out["eigenvalues"]) > 0
        assert np.isclose(sum(pca_out["var_explained"]), 1.0, atol=1e-3)
        
        hypo = perform_seasonal_hypothesis_test(df)
        assert "p_value" in hypo
        assert "winter_mean" in hypo
        
        opt = simulate_optimizers_on_aqi_regression(df, epochs=10)
        assert "Adam" in opt["final_loss"]
        assert opt["final_loss"]["Adam"] < opt["Adam"][0]


