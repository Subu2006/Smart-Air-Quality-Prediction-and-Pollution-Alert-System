"""Smart Air Quality Prediction and Pollution Alert System."""
from .data import load_air_quality_csv, DataLoadResult, POLLUTANTS
from .features import make_forecasting_frame, split_chronologically
from .alerts import aqi_category, evaluate_observation, evaluate_prediction, build_alert_history
from .models import train_model, load_model, make_single_prediction
from .math_foundations import compute_air_quality_pca, perform_seasonal_hypothesis_test, simulate_optimizers_on_aqi_regression

__version__ = "2.0.0"
__all__ = [
    "load_air_quality_csv", "DataLoadResult", "POLLUTANTS",
    "make_forecasting_frame", "split_chronologically",
    "aqi_category", "evaluate_observation", "evaluate_prediction", "build_alert_history",
    "train_model", "load_model", "make_single_prediction",
    "compute_air_quality_pca", "perform_seasonal_hypothesis_test", "simulate_optimizers_on_aqi_regression"
]
