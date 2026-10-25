"""Reproducible model training, chronological evaluation, artifact persistence and inference."""
from __future__ import annotations
from pathlib import Path
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, f1_score
from .data import load_air_quality_csv, write_json
from .features import make_forecasting_frame, split_chronologically
from .alerts import aqi_category

LOGGER = logging.getLogger(__name__)


def _pipeline(model, numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _metrics(y_true, prediction):
    mae = float(mean_absolute_error(y_true, prediction))
    rmse = float(root_mean_squared_error(y_true, prediction))
    r2 = float(r2_score(y_true, prediction))
    
    # Calculate categorical metrics by mapping continuous AQI to CPCB categories
    true_cats = [aqi_category(v) for v in y_true]
    pred_cats = [aqi_category(v) for v in prediction]
    cat_acc = float(accuracy_score(true_cats, pred_cats))
    macro_f1 = float(f1_score(true_cats, pred_cats, average="macro", zero_division=0))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Category_Accuracy": cat_acc,
        "Macro_F1": macro_f1
    }


def train_model(data_path: str | Path, model_path: str | Path, metrics_path: str | Path, *, data_kind: str = "auto", random_state: int = 42) -> dict:
    loaded = load_air_quality_csv(data_path)
    df = loaded.frame
    kind = loaded.audit["data_kind"] if data_kind == "auto" else data_kind
    X, y, target_dates = make_forecasting_frame(df)
    if len(X) < 100:
        raise ValueError(f"Only {len(X)} consecutive-day samples are available; at least 100 are required for a useful demonstration.")
    
    X_train, X_test, y_train, y_test, cutoff = split_chronologically(X, y, target_dates, test_fraction=0.2)
    
    # Validation split inside training set (last 20% of training dates) for unbiased candidate selection
    train_target_dates = target_dates.loc[pd.to_datetime(target_dates) < cutoff].reset_index(drop=True)
    X_fit, X_val, y_fit, y_val, validation_cutoff = split_chronologically(X_train, y_train, train_target_dates, test_fraction=0.2)
    
    categorical_cols = ["City"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    
    model_factories = {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(alpha=10.0),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=random_state),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, max_features=0.85, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state),
    }
    
    # 1. Validation evaluation for hyperparameter/candidate model selection
    validation_rows = []
    for name, factory in model_factories.items():
        selection_pipe = _pipeline(factory(), numeric_cols, categorical_cols)
        selection_pipe.fit(X_fit, y_fit)
        val_pred = selection_pipe.predict(X_val)
        row = _metrics(y_val, val_pred)
        row.update({
            "Model": name,
            "FitRows": int(len(X_fit)),
            "ValidationRows": int(len(X_val)),
            "ValidationStartDate": str(validation_cutoff.date()),
            "SelectionRole": "model selection only"
        })
        validation_rows.append(row)
        
    validation_table = pd.DataFrame(validation_rows).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
    selected_name = str(validation_table.iloc[0]["Model"])

    # 2. Refit all candidates on the full outer training period and evaluate on held-out test set
    test_rows = []
    fitted = {}
    test_predictions = {}
    
    for name, factory in model_factories.items():
        pipe = _pipeline(factory(), numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metric = _metrics(y_test, pred)
        metric.update({
            "Model": name,
            "TrainRows": int(len(X_train)),
            "TestRows": int(len(X_test)),
            "TestStartDate": str(cutoff.date()),
            "SelectedByValidation": name == selected_name
        })
        test_rows.append(metric)
        fitted[name] = pipe
        test_predictions[name] = pred

    # Baseline persistence model: AQI[t] predicts AQI[t+1]
    persistence = X_test["AQI"].fillna(X_train["AQI"].median()).to_numpy()
    persistence_metric = _metrics(y_test, persistence)
    persistence_metric.update({
        "Model": "Persistence baseline (AQI_t)",
        "TrainRows": int(len(X_train)),
        "TestRows": int(len(X_test)),
        "TestStartDate": str(cutoff.date()),
        "SelectedByValidation": False
    })
    test_rows.append(persistence_metric)
    test_predictions["Persistence baseline (AQI_t)"] = persistence
    
    test_table = pd.DataFrame(test_rows).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)

    # Calculate residual standard deviation for selected model (for prediction uncertainty bounds)
    selected_pred = test_predictions[selected_name]
    residuals = y_test.to_numpy() - selected_pred
    residual_std = float(np.std(residuals))
    
    # Feature importances if available
    feature_importances = {}
    final_model = fitted[selected_name].named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        try:
            feature_names = fitted[selected_name].named_steps["preprocess"].get_feature_names_out()
            importances = final_model.feature_importances_
            # Aggregate or take top features
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
            top_fi = fi_df.head(15).to_dict(orient="records")
            feature_importances = {item["Feature"]: round(float(item["Importance"]), 4) for item in top_fi}
        except Exception:
            pass

    # Save artifacts
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected_name], model_path)
    
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(metrics_path.with_suffix(".csv"), index=False)
    
    validation_path = metrics_path.parent / (metrics_path.stem + "_validation.csv")
    validation_table.to_csv(validation_path, index=False)
    
    # Save test predictions table for detailed error analysis & plotting
    test_eval_df = pd.DataFrame({
        "Date": target_dates.loc[X_test.index].values,
        "City": X_test["City"].values,
        "Actual_AQI": y_test.values,
        "Predicted_AQI": selected_pred,
        "Residual": residuals
    })
    test_eval_df.to_csv(metrics_path.parent / (metrics_path.stem + "_test_eval.csv"), index=False)

    metadata = {
        "artifact": str(model_path.name),
        "model_name": selected_name,
        "selection_method": "lowest validation MAE from chronological validation window inside outer training period",
        "target": "AQI at t+1 calendar day",
        "prediction_horizon_days": 1,
        "dataset_kind": kind,
        "dataset_file": Path(data_path).name,
        "data_quality_audit": loaded.audit,
        "rows_used_for_forecasting": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cutoff_date": str(cutoff.date()),
        "cutoff_date_semantics": "first held-out TARGET date; split is grouped chronologically by target date",
        "validation_cutoff_date": str(validation_cutoff.date()),
        "residual_std": residual_std,
        "prediction_interval_margin_95": float(1.96 * residual_std),
        "feature_importances": feature_importances,
        "metric_table_csv": metrics_path.with_suffix(".csv").name,
        "validation_table_csv": validation_path.name,
        "test_eval_csv": (metrics_path.stem + "_test_eval.csv"),
        "model_comparison": test_table.to_dict(orient="records"),
        "validation_model_comparison": validation_table.to_dict(orient="records"),
        "baseline_model": "Persistence: next-day AQI equals current-day AQI",
        "limitations": [
            "The final test period was held out chronologically; model choice was made on an earlier validation period.",
            "A one-day target is available only when the next calendar day's record exists for the same city.",
            "Historical performance is not a guarantee of future accuracy.",
            "Extreme AQI values (>500) exist in raw monitoring records and represent extreme pollution events or sensor saturation.",
            "Predictions are decision-support estimates and not official government alerts."
        ],
        "random_state": random_state,
    }
    write_json(metadata, metrics_path)
    return metadata


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run the training command from README.md.")
    return joblib.load(path)


def make_single_prediction(model, city: str, observation_date, current_values: dict, metadata: dict | None = None) -> dict:
    """Predict next-day AQI from one observed day. Missing pollution values are imputed by fitted pipeline."""
    date = pd.Timestamp(observation_date)
    target_date = date + pd.Timedelta(days=1)
    features = {
        "City": str(city),
        "AQI": current_values.get("AQI"),
        "target_month": target_date.month,
        "target_dayofweek": target_date.dayofweek,
        "target_dayofyear_sin": float(np.sin(2 * np.pi * target_date.dayofyear / 365.25)),
        "target_dayofyear_cos": float(np.cos(2 * np.pi * target_date.dayofyear / 365.25))
    }
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    for col in pollutant_cols:
        if col in current_values:
            try:
                val = current_values[col]
                features[col] = float(val) if val is not None and not pd.isna(val) else np.nan
            except (TypeError, ValueError):
                features[col] = np.nan
        else:
            features[col] = np.nan
            
    X = pd.DataFrame([features])
    value = float(model.predict(X)[0])
    
    # Calculate uncertainty margin
    margin = float(metadata.get("prediction_interval_margin_95", 35.0)) if metadata else 35.0
    lower_bound = max(0.0, float(value - margin))
    upper_bound = float(value + margin)
    category = aqi_category(value)
    
    return {
        "city": str(city),
        "observation_date": str(date.date()),
        "prediction_date": str(target_date.date()),
        "predicted_aqi": round(value, 2),
        "lower_bound_95": round(lower_bound, 2),
        "upper_bound_95": round(upper_bound, 2),
        "category": category,
        "model_name": (metadata or {}).get("model_name", "Random Forest Regressor"),
        "dataset_kind": (metadata or {}).get("dataset_kind", "external_or_user_supplied"),
        "warning": "Model output only; not an official CPCB observation or warning."
    }
