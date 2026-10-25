"""Run complete exploratory and model analysis on real air quality data and generate publication figures."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from smart_air_quality.math_foundations import (
    compute_air_quality_pca,
    perform_seasonal_hypothesis_test,
    simulate_optimizers_on_aqi_regression
)

DATA_PATH = ROOT / "data/raw/city_day.csv"
FIG_DIR = ROOT / "reports/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_METRICS_PATH = ROOT / "models/air_quality_model.csv"
MODEL_JSON_PATH = ROOT / "models/air_quality_model.json"
TEST_EVAL_PATH = ROOT / "models/air_quality_model_test_eval.csv"

# Styling constants
NAVY = "#163447"
TEAL = "#167E78"
CORAL = "#D66B5D"
AMBER = "#F4C66A"
SLATE = "#5E7480"
LIGHT_BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring / Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def run_full_analysis():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City"]).sort_values(["City", "Date"]).reset_index(drop=True)
    
    # 1. Dataset stats
    total_records = len(df)
    cities_count = df["City"].nunique()
    valid_aqi_df = df.dropna(subset=["AQI"]).copy()
    valid_aqi_records = len(valid_aqi_df)
    mean_aqi = float(valid_aqi_df["AQI"].mean())
    median_aqi = float(valid_aqi_df["AQI"].median())
    std_aqi = float(valid_aqi_df["AQI"].std())
    min_aqi = float(valid_aqi_df["AQI"].min())
    max_aqi = float(valid_aqi_df["AQI"].max())
    skew_aqi = float(stats.skew(valid_aqi_df["AQI"]))
    kurt_aqi = float(stats.kurtosis(valid_aqi_df["AQI"]))

    print(f"Total Rows: {total_records}, Valid AQI Rows: {valid_aqi_records}")
    print(f"AQI Stats: Mean={mean_aqi:.1f}, Median={median_aqi:.1f}, Min={min_aqi:.1f}, Max={max_aqi:.1f}, Skew={skew_aqi:.2f}, Kurtosis={kurt_aqi:.2f}")

    # Add Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["Season"] = df["Month"].apply(get_season)

    # -------------------------------------------------------------
    # Figure 1: AQI Distribution & Boxplot
    # -------------------------------------------------------------
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]})
    fig.patch.set_facecolor("white")
    ax_box.set_facecolor(LIGHT_BG)
    ax_hist.set_facecolor(LIGHT_BG)

    # Boxplot
    sns.boxplot(x=valid_aqi_df["AQI"], ax=ax_box, color=TEAL, fliersize=2.5, boxprops=dict(alpha=0.85))
    ax_box.set(xlabel="")
    ax_box.set_title("National Air Quality Index (AQI) Statistical Distribution (2015–2020)", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax_box.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)

    # Histogram + KDE
    sns.histplot(valid_aqi_df["AQI"], kde=True, ax=ax_hist, color=NAVY, bins=60, alpha=0.6, stat="density")
    ax_hist.axvline(mean_aqi, color=CORAL, linestyle="--", linewidth=2, label=f"Mean: {mean_aqi:.1f}")
    ax_hist.axvline(median_aqi, color=AMBER, linestyle="-.", linewidth=2, label=f"Median: {median_aqi:.1f}")

    # Shade CPCB Bands
    cpcb_bands = [
        (0, 50, "Good", "#10B981"),
        (50, 100, "Satisfactory", "#84CC16"),
        (100, 200, "Moderate", "#FBBF24"),
        (200, 300, "Poor", "#F97316"),
        (300, 400, "Very Poor", "#EF4444"),
        (400, 500, "Severe", "#7F1D1D")
    ]
    for low, high, label, color in cpcb_bands:
        ax_hist.axvspan(low, high, color=color, alpha=0.1)

    ax_hist.set_xlabel("Air Quality Index (AQI)", fontsize=11, fontweight="bold", color=NAVY)
    ax_hist.set_ylabel("Probability Density", fontsize=11, fontweight="bold", color=NAVY)
    ax_hist.set_xlim(0, 600)
    ax_hist.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID_COLOR)

    # Statistical text annotation box
    stat_text = (
        f"Statistical Moments:\n"
        f"• Sample Size: {valid_aqi_records:,}\n"
        f"• Mean: {mean_aqi:.1f}\n"
        f"• Median: {median_aqi:.1f}\n"
        f"• Std Dev: {std_aqi:.1f}\n"
        f"• Skewness: {skew_aqi:.2f} (Right-skewed)\n"
        f"• Kurtosis: {kurt_aqi:.2f} (Leptokurtic)\n"
        f"• Extreme Rows (>500): {(valid_aqi_df['AQI'] > 500).sum():,}"
    )
    ax_hist.text(0.72, 0.40, stat_text, transform=ax_hist.transAxes, fontsize=9.5,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=SLATE, alpha=0.9))

    plt.tight_layout()
    fig1_path = FIG_DIR / "fig01_national_aqi_distribution.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path.name}")

    # -------------------------------------------------------------
    # Figure 2: Pollutant Pearson Correlation Heatmap
    # -------------------------------------------------------------
    pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI"]
    corr_df = df[pollutants].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(LIGHT_BG)

    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
        annot_kws={"size": 9, "weight": "bold"}
    )
    ax.set_title("Inter-Pollutant Pearson Correlation Matrix (Criteria Pollutants vs AQI)", fontsize=12, fontweight="bold", color=NAVY, pad=15)
    plt.tight_layout()
    fig2_path = FIG_DIR / "fig02_pollutant_correlation_matrix.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path.name}")

    # -------------------------------------------------------------
    # Figure 3: Temporal Trends: Monthly Seasonality & Year-over-Year
    # -------------------------------------------------------------
    fig, (ax_month, ax_season) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("white")
    ax_month.set_facecolor(LIGHT_BG)
    ax_season.set_facecolor(LIGHT_BG)

    # Monthly line by year
    years_to_plot = [2016, 2017, 2018, 2019]
    monthly_year_df = df[df["Year"].isin(years_to_plot)].groupby(["Year", "Month"])["AQI"].mean().reset_index()
    palette = [NAVY, TEAL, AMBER, CORAL]
    for yr, color in zip(years_to_plot, palette):
        sub = monthly_year_df[monthly_year_df["Year"] == yr]
        ax_month.plot(sub["Month"], sub["AQI"], marker="o", linewidth=2.0, label=str(yr), color=color)

    ax_month.set_title("Monthly AQI Trajectory Across Calendar Years (2016–2019)", fontsize=11, fontweight="bold", color=NAVY)
    ax_month.set_xlabel("Month of Year", fontsize=10, fontweight="bold", color=NAVY)
    ax_month.set_ylabel("Mean Air Quality Index (AQI)", fontsize=10, fontweight="bold", color=NAVY)
    ax_month.set_xticks(range(1, 13))
    ax_month.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax_month.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
    ax_month.legend(title="Year", frameon=True, facecolor="white")

    # Seasonal Bar Chart with error bars
    season_order = ["Winter", "Post-Monsoon", "Spring / Pre-Monsoon", "Monsoon"]
    season_stats = df.groupby("Season")["AQI"].agg(["mean", "std", "count"]).reindex(season_order)
    season_colors = ["#3B82F6", "#F97316", "#FBBF24", "#10B981"]

    ax_season.set_xticks(range(len(season_stats)))
    bars = ax_season.bar(range(len(season_stats)), season_stats["mean"], yerr=season_stats["std"] / np.sqrt(season_stats["count"]),
                         capsize=5, color=season_colors, alpha=0.85, edgecolor=NAVY)
    
    for bar in bars:
        h = bar.get_height()
        ax_season.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax_season.set_title("Seasonal Mean AQI (± SEM)", fontsize=11, fontweight="bold", color=NAVY)
    ax_season.set_ylabel("Mean AQI", fontsize=10, fontweight="bold", color=NAVY)
    ax_season.set_xticklabels(["Winter\n(Dec-Feb)", "Post-Monsoon\n(Oct-Nov)", "Pre-Monsoon\n(Mar-May)", "Monsoon\n(Jun-Sep)"], fontsize=8.5)
    ax_season.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="y")

    plt.tight_layout()
    fig3_path = FIG_DIR / "fig03_temporal_monthly_seasonal.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path.name}")

    # -------------------------------------------------------------
    # Figure 4: City Rankings (Top 10 Most Polluted vs Bottom 5)
    # -------------------------------------------------------------
    city_summary = df.groupby("City")["AQI"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    top10 = city_summary.head(10).reset_index()
    bottom5 = city_summary.tail(5).sort_values("mean", ascending=True).reset_index()

    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("white")
    ax_top.set_facecolor(LIGHT_BG)
    ax_bot.set_facecolor(LIGHT_BG)

    # Top 10 polluted
    bars_top = ax_top.barh(top10["City"][::-1], top10["mean"][::-1], color=CORAL, edgecolor=NAVY, alpha=0.85)
    for bar in bars_top:
        w = bar.get_width()
        ax_top.annotate(f"{w:.1f}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
    ax_top.set_title("Top 10 Most Polluted Indian Cities (Mean AQI 2015–2020)", fontsize=11, fontweight="bold", color=NAVY)
    ax_top.set_xlabel("Average AQI", fontsize=10, fontweight="bold", color=NAVY)
    ax_top.set_xlim(0, max(top10["mean"]) * 1.15)
    ax_top.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="x")

    # Bottom 5 cleanest
    bars_bot = ax_bot.barh(bottom5["City"], bottom5["mean"], color="#10B981", edgecolor=NAVY, alpha=0.85)
    for bar in bars_bot:
        w = bar.get_width()
        ax_bot.annotate(f"{w:.1f}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
    ax_bot.set_title("Top 5 Cleanest Monitored Cities (Mean AQI)", fontsize=11, fontweight="bold", color=NAVY)
    ax_bot.set_xlabel("Average AQI", fontsize=10, fontweight="bold", color=NAVY)
    ax_bot.set_xlim(0, max(bottom5["mean"]) * 1.25)
    ax_bot.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="x")

    plt.tight_layout()
    fig4_path = FIG_DIR / "fig04_city_pollution_rankings.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path.name}")

    # -------------------------------------------------------------
    # Figure 5: Model Performance Benchmarking
    # -------------------------------------------------------------
    if MODEL_METRICS_PATH.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_PATH)
        fig, (ax_mae, ax_r2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("white")
        ax_mae.set_facecolor(LIGHT_BG)
        ax_r2.set_facecolor(LIGHT_BG)

        models = metrics_df["Model"].tolist()
        mae_vals = metrics_df["MAE"].tolist()
        r2_vals = metrics_df["R2"].tolist()
        colors = [TEAL if m == "Random Forest" else SLATE for m in models]

        ax_mae.set_xticks(range(len(models)))
        bars1 = ax_mae.bar(range(len(models)), mae_vals, color=colors, edgecolor=NAVY, alpha=0.85)
        for bar in bars1:
            h = bar.get_height()
            ax_mae.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_mae.set_title("Mean Absolute Error (MAE) on Chronological Test Split", fontsize=11, fontweight="bold", color=NAVY)
        ax_mae.set_ylabel("MAE (AQI Points, lower is better)", fontsize=10, fontweight="bold", color=NAVY)
        ax_mae.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
        ax_mae.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="y")

        ax_r2.set_xticks(range(len(models)))
        bars2 = ax_r2.bar(range(len(models)), r2_vals, color=colors, edgecolor=NAVY, alpha=0.85)
        for bar in bars2:
            h = bar.get_height()
            ax_r2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                           textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_r2.set_title("Explained Variance (R² Score) on Out-of-Sample Test Split", fontsize=11, fontweight="bold", color=NAVY)
        ax_r2.set_ylabel("R² (higher is better)", fontsize=10, fontweight="bold", color=NAVY)
        ax_r2.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
        ax_r2.set_ylim(0.65, 0.90)
        ax_r2.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="y")

        plt.tight_layout()
        fig5_path = FIG_DIR / "fig05_model_performance_comparison.png"
        plt.savefig(fig5_path, dpi=200)
        plt.close()
        print(f"Saved: {fig5_path.name}")

    # -------------------------------------------------------------
    # Figure 6: Actual vs Predicted Parity & Residual Distribution
    # -------------------------------------------------------------
    if TEST_EVAL_PATH.exists():
        eval_df = pd.read_csv(TEST_EVAL_PATH)
        eval_sample = eval_df.sample(min(2500, len(eval_df)), random_state=42)

        fig, (ax_parity, ax_res) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("white")
        ax_parity.set_facecolor(LIGHT_BG)
        ax_res.set_facecolor(LIGHT_BG)

        # Parity plot
        ax_parity.scatter(eval_sample["Actual_AQI"], eval_sample["Predicted_AQI"], alpha=0.25, color=TEAL, s=18)
        max_val = min(500, max(eval_sample["Actual_AQI"].quantile(0.99), eval_sample["Predicted_AQI"].quantile(0.99)))
        ax_parity.plot([0, max_val], [0, max_val], color=CORAL, linestyle="--", linewidth=2.0, label="1:1 Perfect Prediction")
        ax_parity.set_title("Random Forest Parity Plot: Actual vs Predicted AQI", fontsize=11, fontweight="bold", color=NAVY)
        ax_parity.set_xlabel("Actual Next-Day AQI", fontsize=10, fontweight="bold", color=NAVY)
        ax_parity.set_ylabel("Predicted Next-Day AQI", fontsize=10, fontweight="bold", color=NAVY)
        ax_parity.set_xlim(0, max_val)
        ax_parity.set_ylim(0, max_val)
        ax_parity.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
        ax_parity.legend(frameon=True, facecolor="white")

        # Residual Distribution
        res = eval_df["Residual"].dropna()
        sns.histplot(res, kde=True, ax=ax_res, color=NAVY, bins=60, alpha=0.6, stat="density")
        ax_res.axvline(0, color=CORAL, linestyle="--", linewidth=2, label="Zero Bias Line")
        ax_res.axvline(float(res.mean()), color=AMBER, linestyle="-.", linewidth=2, label=f"Mean Error: {res.mean():.2f}")
        ax_res.set_title("Forecast Residual Distribution (Error = Actual - Predicted)", fontsize=11, fontweight="bold", color=NAVY)
        ax_res.set_xlabel("Residual Error (AQI Points)", fontsize=10, fontweight="bold", color=NAVY)
        ax_res.set_xlim(-150, 150)
        ax_res.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
        ax_res.legend(frameon=True, facecolor="white")

        plt.tight_layout()
        fig6_path = FIG_DIR / "fig06_predicted_vs_actual_residuals.png"
        plt.savefig(fig6_path, dpi=200)
        plt.close()
        print(f"Saved: {fig6_path.name}")

    # -------------------------------------------------------------
    # Figure 7: Feature Importances
    # -------------------------------------------------------------
    if MODEL_JSON_PATH.exists():
        meta = json.loads(MODEL_JSON_PATH.read_text())
        fi = meta.get("feature_importances", {})
        if fi:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("white")
            ax.set_facecolor(LIGHT_BG)
            
            fi_series = pd.Series(fi).sort_values(ascending=True)
            clean_names = [f.replace("numeric__", "").replace("categorical__City_", "City: ") for f in fi_series.index]
            
            bars = ax.barh(clean_names, fi_series.values, color=TEAL, edgecolor=NAVY, alpha=0.85)
            for bar in bars:
                w = bar.get_width()
                ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")
            
            ax.set_title("Feature Importance Ranking (Gini Impurity Reduction)", fontsize=11, fontweight="bold", color=NAVY)
            ax.set_xlabel("Normalized Relative Importance", fontsize=10, fontweight="bold", color=NAVY)
            ax.set_xlim(0, max(fi_series.values) * 1.2)
            ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="x")
            plt.tight_layout()
            fig7_path = FIG_DIR / "fig07_feature_importance.png"
            plt.savefig(fig7_path, dpi=200)
            plt.close()
            print(f"Saved: {fig7_path.name}")

    # -------------------------------------------------------------
    # Figure 8: CPCB Category Breakdown Donut Chart
    # -------------------------------------------------------------
    cat_counts = valid_aqi_df["AQI_Bucket"].value_counts()
    cpcb_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    ordered_counts = cat_counts.reindex(cpcb_order).dropna()
    cpcb_palette = ["#10B981", "#84CC16", "#FBBF24", "#F97316", "#EF4444", "#7F1D1D"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor("white")
    wedges, texts, autotexts = ax.pie(
        ordered_counts.values,
        labels=ordered_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=cpcb_palette,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    plt.setp(autotexts, size=9.5, weight="bold", color="white")
    plt.setp(texts, size=10, color=NAVY, weight="bold")
    ax.set_title("Proportion of Monitoring Days by National AQI Category (2015–2020)", fontsize=11, fontweight="bold", color=NAVY, pad=15)
    ax.text(0, 0, f"Total\n{valid_aqi_records:,}\nObservations", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    
    plt.tight_layout()
    fig8_path = FIG_DIR / "fig08_cpcb_category_distribution.png"
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f"Saved: {fig8_path.name}")

    # -------------------------------------------------------------
    # Figure 9: Mathematical Foundations: PCA Scree & Component Loadings
    # -------------------------------------------------------------
    pca_res = compute_air_quality_pca(df, ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])
    if "error" not in pca_res:
        fig, (ax_scree, ax_load) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("white")
        ax_scree.set_facecolor(LIGHT_BG)
        ax_load.set_facecolor(LIGHT_BG)

        # Scree plot
        pcs = [f"PC{i+1}" for i in range(len(pca_res["eigenvalues"]))]
        var_pcts = [v * 100 for v in pca_res["var_explained"]]
        cum_pcts = [v * 100 for v in pca_res["cum_var_explained"]]

        ax_scree.bar(pcs, var_pcts, color=TEAL, alpha=0.8, edgecolor=NAVY, label="Individual %")
        ax_scree.plot(pcs, cum_pcts, color=CORAL, marker="o", linewidth=2, label="Cumulative %")
        for i, (v, c) in enumerate(zip(var_pcts, cum_pcts)):
            ax_scree.annotate(f"{v:.1f}%", xy=(i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
            ax_scree.annotate(f"{c:.1f}%", xy=(i, c), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=CORAL, fontweight="bold")

        ax_scree.set_title("Principal Component Analysis: Scree Plot (Eigenvalue Variance)", fontsize=11, fontweight="bold", color=NAVY)
        ax_scree.set_ylabel("Variance Explained (%)", fontsize=10, fontweight="bold", color=NAVY)
        ax_scree.set_ylim(0, 115)
        ax_scree.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR, axis="y")
        ax_scree.legend(loc="center right", frameon=True, facecolor="white")

        # Loadings Biplot
        loadings_df = pd.DataFrame(pca_res["loadings"])
        ax_load.scatter(loadings_df["PC1_Loading"], loadings_df["PC2_Loading"], color=NAVY, s=60, zorder=3)
        for _, r in loadings_df.iterrows():
            ax_load.arrow(0, 0, r["PC1_Loading"], r["PC2_Loading"], color=TEAL, alpha=0.7, head_width=0.03, zorder=2)
            ax_load.annotate(r["Pollutant"], xy=(r["PC1_Loading"], r["PC2_Loading"]), xytext=(4, 4), textcoords="offset points",
                             fontsize=9.5, fontweight="bold", color=NAVY)

        ax_load.axhline(0, color=SLATE, linestyle="--", alpha=0.4)
        ax_load.axvline(0, color=SLATE, linestyle="--", alpha=0.4)
        ax_load.set_title("Pollutant Factor Loadings on PC1 & PC2", fontsize=11, fontweight="bold", color=NAVY)
        ax_load.set_xlabel("PC1 (Combustion / Particulate Axis)", fontsize=10, fontweight="bold", color=NAVY)
        ax_load.set_ylabel("PC2 (Photochemical / Gaseous Axis)", fontsize=10, fontweight="bold", color=NAVY)
        ax_load.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)

        plt.tight_layout()
        fig9_path = FIG_DIR / "fig09_pca_scree_and_biplot.png"
        plt.savefig(fig9_path, dpi=200)
        plt.close()
        print(f"Saved: {fig9_path.name}")

    # -------------------------------------------------------------
    # Figure 10: Optimization Loss Trajectories (SGD vs Momentum vs Adam)
    # -------------------------------------------------------------
    opt_res = simulate_optimizers_on_aqi_regression(df, epochs=50, learning_rate=0.03)
    fig, ax_opt = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white")
    ax_opt.set_facecolor(LIGHT_BG)

    opt_palette = {"BGD": NAVY, "SGD": SLATE, "Momentum": AMBER, "RMSprop": "#8B5CF6", "Adam": TEAL}
    for opt_name, color in opt_palette.items():
        lw = 2.2 if opt_name in ["Adam", "Momentum"] else 1.5
        ax_opt.plot(opt_res["epochs"], opt_res[opt_name], label=f"{opt_name} (final: {opt_res['final_loss'][opt_name]:.4f})",
                    color=color, linewidth=lw)

    ax_opt.set_title("Comparison of Gradient Descent Optimization Algorithms on AQI Loss", fontsize=11, fontweight="bold", color=NAVY)
    ax_opt.set_xlabel("Optimization Epochs", fontsize=10, fontweight="bold", color=NAVY)
    ax_opt.set_ylabel("Mean Squared Regression Loss", fontsize=10, fontweight="bold", color=NAVY)
    ax_opt.set_yscale("log")
    ax_opt.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
    ax_opt.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    fig10_path = FIG_DIR / "fig10_optimizer_convergence_curves.png"
    plt.savefig(fig10_path, dpi=200)
    plt.close()
    print(f"Saved: {fig10_path.name}")

    # -------------------------------------------------------------
    # Statistical Hypothesis Testing
    # -------------------------------------------------------------
    hypo_res = perform_seasonal_hypothesis_test(df)

    # -------------------------------------------------------------
    # Write summary JSON for report generation and app metrics
    # -------------------------------------------------------------
    summary = {
        "dataset": {
            "total_records": total_records,
            "valid_aqi_records": valid_aqi_records,
            "cities_count": cities_count,
            "start_date": str(df["Date"].min().date()),
            "end_date": str(df["Date"].max().date()),
            "aqi_mean": round(mean_aqi, 2),
            "aqi_median": round(median_aqi, 2),
            "aqi_std": round(std_aqi, 2),
            "aqi_min": round(min_aqi, 2),
            "aqi_max": round(max_aqi, 2),
            "aqi_skewness": round(skew_aqi, 2),
            "aqi_kurtosis": round(kurt_aqi, 2),
            "missing_aqi_count": int(df["AQI"].isna().sum()),
            "extreme_aqi_count": int((df["AQI"] > 500).sum())
        },
        "top_cities_avg_aqi": {c: round(float(m), 2) for c, m in zip(top10["City"], top10["mean"])},
        "cleanest_cities_avg_aqi": {c: round(float(m), 2) for c, m in zip(bottom5["City"], bottom5["mean"])},
        "seasonal_means": {s: round(float(m), 2) for s, m in zip(season_stats.index, season_stats["mean"])},
        "category_distribution": {c: int(v) for c, v in zip(ordered_counts.index, ordered_counts.values)},
        "hypothesis_testing": hypo_res,
        "pca_summary": {
            "pc1_var": round(float(pca_res["var_explained"][0] * 100), 2) if "error" not in pca_res else 0,
            "pc2_var": round(float(pca_res["var_explained"][1] * 100), 2) if "error" not in pca_res else 0,
            "cum_2pc_var": round(float(pca_res["cum_var_explained"][1] * 100), 2) if "error" not in pca_res else 0,
        },
        "optimizer_final_loss": opt_res["final_loss"]
    }
    
    summary_path = ROOT / "reports/results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Analysis complete. Results summary saved to: {summary_path}")


if __name__ == "__main__":
    run_full_analysis()
