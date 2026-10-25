"""Train and evaluate reproducible one-day-ahead AQI models on real or demo datasets."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from smart_air_quality.models import train_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None, help="CSV path with Date, City, AQI, and pollutant columns")
    p.add_argument("--model", default="models/air_quality_model.joblib", help="Output model artifact path")
    p.add_argument("--metadata", default=None, help="Output JSON metadata path")
    p.add_argument("--data-kind", choices=["auto", "real", "synthetic_demo"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # Priority: specified path -> data/raw/city_day.csv -> data/demo/air_quality_demo.csv
    if a.data:
        data = Path(a.data)
    elif (ROOT / "data/raw/city_day.csv").exists():
        data = ROOT / "data/raw/city_day.csv"
    else:
        data = ROOT / "data/demo/air_quality_demo.csv"

    model = Path(a.model)
    model = model if model.is_absolute() else ROOT / model
    
    metadata = Path(a.metadata) if a.metadata else model.with_suffix(".json")
    metadata = metadata if metadata.is_absolute() else ROOT / metadata
    
    print(f"Training on dataset: {data} (kind: {a.data_kind})")
    metrics = train_model(data, model, metadata, data_kind=a.data_kind, random_state=a.seed)
    
    print("\n--- Training & Chronological Evaluation Summary ---")
    print(json.dumps({
        k: metrics[k] for k in [
            "model_name", "target", "prediction_horizon_days", "dataset_file",
            "rows_used_for_forecasting", "train_rows", "test_rows", "cutoff_date",
            "residual_std", "prediction_interval_margin_95"
        ]
    }, indent=2))
    
    print("\nCandidate Model Performance (Chronological Outer Test Split):")
    comparison = metrics.get("model_comparison", [])
    print(f"{'Model':<30} {'MAE':<8} {'RMSE':<8} {'R2':<8} {'Cat Acc':<10} {'Macro F1':<10}")
    print("-" * 76)
    for row in comparison:
        print(f"{row['Model']:<30} {row['MAE']:<8.2f} {row['RMSE']:<8.2f} {row['R2']:<8.4f} {row.get('Category_Accuracy', 0):<10.4f} {row.get('Macro_F1', 0):<10.4f}")
    
    print(f"\nArtifact saved: {model}")
    print(f"Metadata saved: {metadata}")
    print(f"Test comparison CSV: {metadata.with_suffix('.csv')}")

if __name__ == "__main__":
    main()
