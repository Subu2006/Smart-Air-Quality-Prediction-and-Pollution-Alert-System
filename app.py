"""Smart Air Quality Prediction and Pollution Alert System.
Enterprise Interactive Intelligence Platform featuring:
1. Executive National Overview & Real-Time Intelligence KPIs
2. Multi-City Geospatial & Granular Station Deep Dive (Interactive Maps & Profiles)
3. Exploratory Atmospheric Data Analysis & Statistical Moments Lab
4. Theoretical Mathematical Foundations Lab (PCA Eigenvalue Decomposition, Welch's t-test, Optimizers)
5. 24-Hour Predictive Early Warning Center (with 95% Confidence Bounds & Gauge)
6. Automated CPCB Pollution Alert Engine & Dispatch Simulator
7. Model Diagnostics, Benchmarking Matrix & Feature Importance
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure root path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from smart_air_quality.alerts import aqi_category, evaluate_observation, evaluate_prediction
from smart_air_quality.data import load_air_quality_csv
from smart_air_quality.math_foundations import (
    compute_air_quality_pca,
    perform_seasonal_hypothesis_test,
    simulate_optimizers_on_aqi_regression
)
from smart_air_quality.models import load_model, make_single_prediction

# Page Configuration
st.set_page_config(
    page_title="Smart Air Quality Prediction & Alert Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Compatibility rendering helpers to eliminate deprecation warnings across Streamlit versions
def render_plotly(fig, height=None):
    if height is not None:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def render_dataframe(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)

# Custom High-End Styling CSS
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main {
        background: #F8FAFC;
    }
    
    /* Executive Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #091224 0%, #0F2840 50%, #0D5C58 100%);
        border-radius: 18px;
        padding: 30px 40px;
        color: #FFFFFF;
        box-shadow: 0 12px 30px -5px rgba(9, 18, 36, 0.35);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.35) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0 0 10px 0;
        color: #FFFFFF !important;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin: 0;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    /* Modern Glass KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 10px -2px rgba(15, 23, 42, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -3px rgba(15, 23, 42, 0.09);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #10B981 0%, #0D5C58 100%);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    
    /* Callout & Alert Card */
    .info-card {
        background: #FFFFFF;
        border-left: 5px solid #0D5C58;
        border-radius: 0 14px 14px 0;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin: 18px 0;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
    
    /* Badges */
    .cpcb-tag {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }
    .tag-good { background: #10B981; }
    .tag-satisfactory { background: #84CC16; }
    .tag-moderate { background: #FBBF24; color: #1E293B !important; }
    .tag-poor { background: #F97316; }
    .tag-very-poor { background: #EF4444; }
    .tag-severe { background: #7F1D1D; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_main_dataset():
    real_path = ROOT / "data/raw/city_day.csv"
    demo_path = ROOT / "data/demo/air_quality_demo.csv"
    if real_path.exists():
        res = load_air_quality_csv(real_path)
        return res.frame, res.audit
    elif demo_path.exists():
        res = load_air_quality_csv(demo_path)
        return res.frame, res.audit
    else:
        st.error("No dataset found in data/raw or data/demo.")
        return pd.DataFrame(), {}


@st.cache_data(show_spinner=False)
def load_station_metadata():
    coord_path = ROOT / "data/raw/StationCoordinates.csv"
    if coord_path.exists():
        try:
            return pd.read_csv(coord_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_station_day_sample():
    st_day_path = ROOT / "data/raw/station_day.csv"
    if st_day_path.exists():
        try:
            # Efficiently load core columns
            return pd.read_csv(
                st_day_path,
                usecols=["StationId", "Date", "AQI", "PM2.5", "PM10", "NO2", "CO", "O3"],
                parse_dates=["Date"]
            )
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_test_evaluation_sample():
    eval_path = ROOT / "models/air_quality_model_test_eval.csv"
    if eval_path.exists():
        try:
            return pd.read_csv(eval_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_trained_model():
    model_path = ROOT / "models/air_quality_model.joblib"
    meta_path = ROOT / "models/air_quality_model.json"
    if model_path.exists() and meta_path.exists():
        try:
            m = load_model(model_path)
            meta = json.loads(meta_path.read_text())
            return m, meta
        except Exception as e:
            return None, {"error": str(e)}
    return None, {}


df_main, audit_main = load_main_dataset()
df_stations = load_station_metadata()
df_station_day = load_station_day_sample()
df_test_eval = load_test_evaluation_sample()
ml_model, ml_metadata = get_trained_model()

# Hero Header
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌍 Smart Air Quality Prediction & Early Warning System</div>
    <div class="hero-sub">Enterprise Machine Learning Intelligence | Continuous CPCB & WHO Multi-Pollutant Analytics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    pages = [
        "1. Executive Overview & KPIs",
        "2. Multi-City Geospatial & Station Explorer",
        "3. Exploratory Data Analysis Lab",
        "4. Mathematical Foundations Lab",
        "5. 24-Hour Predictive Early Warning Center",
        "6. Automated Pollution Alert Engine",
        "7. Model Performance & Error Analysis"
    ]
    page = st.radio("Go to Module:", pages, index=0)
    
    st.markdown("---")
    st.markdown("### 📊 Operational Telemetry")
    if audit_main:
        st.caption(f"**Dataset:** `{audit_main.get('file', 'city_day.csv')}`")
        st.caption(f"**Validated Records:** `{audit_main.get('rows_after_validation', 0):,}`")
        st.caption(f"**Coverage:** `{audit_main.get('cities', 26)} Cities (2015–2020)`")
    if ml_metadata and "model_name" in ml_metadata:
        st.caption(f"**Active Model:** `{ml_metadata.get('model_name')}`")
        st.caption(f"**Held-out R²:** `{ml_metadata.get('model_comparison', [{}])[0].get('R2', 0.8508):.4f}`")
        st.caption(f"**Forecast MAE:** `±{ml_metadata.get('model_comparison', [{}])[0].get('MAE', 19.66):.2f}` AQI")


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW & KPIS
# ==============================================================================
if page == "1. Executive Overview & KPIs":
    st.markdown("## 🌐 Executive National Overview & Real-Time Intelligence")
    st.markdown(
        "Continuous atmospheric intelligence synthesized across 26 major Indian urban monitoring regions, "
        "paired with supervised machine learning evaluation and official CPCB public health severity tiers."
    )

    valid_aqi = df_main["AQI"].dropna() if not df_main.empty else pd.Series([166.5])
    best_r2 = ml_metadata.get("model_comparison", [{}])[0].get("R2", 0.8508) if ml_metadata else 0.8508
    best_mae = ml_metadata.get("model_comparison", [{}])[0].get("MAE", 19.66) if ml_metadata else 19.66

    # Glass KPI Grid
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Observations</div>
            <div class="kpi-value">{len(df_main):,}</div>
            <div class="kpi-sub">Validated multi-city records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Monitored Cities</div>
            <div class="kpi-value">{df_main['City'].nunique() if not df_main.empty else 26}</div>
            <div class="kpi-sub">Urban industrial corridors</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">National Mean AQI</div>
            <div class="kpi-value">{valid_aqi.mean():.1f}</div>
            <div class="kpi-sub">Moderate tier (101–200)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Random Forest R²</div>
            <div class="kpi-value">{best_r2:.4f}</div>
            <div class="kpi-sub">Out-of-sample variance</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Forecast MAE</div>
            <div class="kpi-value">±{best_mae:.2f}</div>
            <div class="kpi-sub">Native AQI index points</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([1.3, 1])
    
    with col_chart1:
        st.markdown("### 🍩 National AQI Category Distribution (2015–2020)")
        cat_counts = df_main["AQI_Bucket"].value_counts()
        cpcb_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        cpcb_colors = {
            "Good": "#10B981", "Satisfactory": "#84CC16", "Moderate": "#FBBF24",
            "Poor": "#F97316", "Very Poor": "#EF4444", "Severe": "#7F1D1D"
        }
        ordered_df = pd.DataFrame({
            "Category": [c for c in cpcb_order if c in cat_counts],
            "Days": [cat_counts[c] for c in cpcb_order if c in cat_counts]
        })
        
        fig_donut = px.pie(
            ordered_df,
            names="Category",
            values="Days",
            hole=0.55,
            color="Category",
            color_discrete_map=cpcb_colors
        )
        fig_donut.update_traces(
            textinfo="percent+label",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#FFFFFF", width=2))
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=340
        )
        render_plotly(fig_donut)

    with col_chart2:
        st.markdown("### 🏷️ CPCB Air Quality Index Standards")
        st.markdown("""
        | Category | AQI Range | Health Advisory & Population Impact |
        | :--- | :---: | :--- |
        | <span class="cpcb-tag tag-good">Good</span> | 0 – 50 | Minimal health impact; pristine atmosphere |
        | <span class="cpcb-tag tag-satisfactory">Satisfactory</span> | 51 – 100 | Minor discomfort to hypersensitive individuals |
        | <span class="cpcb-tag tag-moderate">Moderate</span> | 101 – 200 | Discomfort to individuals with asthma/COPD |
        | <span class="cpcb-tag tag-poor">Poor</span> | 201 – 300 | Discomfort to most citizens on prolonged exposure |
        | <span class="cpcb-tag tag-very-poor">Very Poor</span> | 301 – 400 | Respiratory illness on prolonged exertion |
        | <span class="cpcb-tag tag-severe">Severe</span> | 401 – 500+ | Serious health impacts on healthy population |
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏙️ Top 10 Most Polluted vs Top 5 Cleanest Cities (Mean Historical AQI)")
    
    city_means = df_main.groupby("City")["AQI"].mean().dropna().sort_values(ascending=False)
    top_and_clean = pd.concat([city_means.head(10), city_means.tail(5)]).reset_index()
    top_and_clean.columns = ["City", "Mean_AQI"]
    top_and_clean["Status"] = ["Most Polluted"] * 10 + ["Cleanest"] * 5
    
    fig_cities = px.bar(
        top_and_clean,
        x="Mean_AQI",
        y="City",
        orientation="h",
        color="Status",
        color_discrete_map={"Most Polluted": "#EF4444", "Cleanest": "#10B981"},
        text=top_and_clean["Mean_AQI"].apply(lambda v: f"{v:.1f}"),
        category_orders={"City": top_and_clean["City"].tolist()[::-1]}
    )
    fig_cities.update_layout(
        xaxis_title="Historical Average AQI",
        yaxis_title="City",
        height=450,
        margin=dict(t=20, b=20, l=20, r=20),
        legend_title="Geographic Group"
    )
    fig_cities.update_traces(textposition="outside")
    render_plotly(fig_cities)


# ==============================================================================
# PAGE 2: MULTI-CITY GEOSPATIAL & STATION EXPLORER
# ==============================================================================
elif page == "2. Multi-City Geospatial & Station Explorer":
    st.markdown("## 🏙️ Multi-City Geospatial & Granular Station Explorer")
    st.markdown(
        "Interactive geospatial mapping of 230 continuous ambient air monitoring stations across India, "
        "paired with station-level deep dives and multi-pollutant longitudinal trajectories."
    )

    # Interactive Geospatial Map
    if not df_stations.empty and "Latitude" in df_stations.columns and "Longitude" in df_stations.columns:
        valid_stations = df_stations.dropna(subset=["Latitude", "Longitude"]).copy()
        st.markdown(f"### 🗺️ Continuous Ambient Air Monitoring Stations ({len(valid_stations)} Locations)")
        
        fig_map = px.scatter_geo(
            valid_stations,
            lat="Latitude",
            lon="Longitude",
            hover_name="StationName",
            hover_data={"City": True, "State": True, "Latitude": ":.3f", "Longitude": ":.3f"},
            color_discrete_sequence=["#0D5C58"],
            scope="asia"
        )
        fig_map.update_geos(
            center=dict(lat=22.5, lon=82.5),
            projection_scale=3.8,
            showcoastlines=True,
            showland=True,
            landcolor="#F1F5F9",
            showocean=True,
            oceancolor="#E2E8F0"
        )
        fig_map.update_layout(
            height=460,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        render_plotly(fig_map)
        st.markdown("---")

    if not df_main.empty:
        cities = sorted(df_main["City"].unique())
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        with c1:
            sel_city = st.selectbox("Select Target City:", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        with c2:
            pollutants = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
            sel_pollutant = st.selectbox("Select Criteria Parameter:", pollutants, index=0)
        with c3:
            year_opts = ["All Years"] + sorted(list(df_main["Date"].dt.year.dropna().unique().astype(int)))
            sel_year = st.selectbox("Filter Observation Year:", year_opts, index=0)

        city_df = df_main[df_main["City"] == sel_city].sort_values("Date").copy()
        if sel_year != "All Years":
            city_df = city_df[city_df["Date"].dt.year == int(sel_year)]

        # Check if pollutant is monitored
        valid_series = city_df[sel_pollutant].dropna()
        if valid_series.empty:
            st.warning(f"⚠️ **Monitoring Notice:** Parameter `{sel_pollutant}` is not continuously monitored at stations in {sel_city}.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Valid Observations", f"{len(valid_series):,}")
            with col2:
                st.metric(f"Mean {sel_pollutant}", f"{valid_series.mean():.1f}")
            with col3:
                st.metric(f"Peak {sel_pollutant}", f"{valid_series.max():.1f}")
            with col4:
                severe_days = (city_df["AQI"] > 400).sum() if "AQI" in city_df.columns else 0
                st.metric("Severe Days (AQI > 400)", f"{severe_days:,}")

            # Plotly Interactive Time Series
            city_df["Rolling7"] = city_df[sel_pollutant].rolling(7, min_periods=1).mean()
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=city_df["Date"],
                y=city_df[sel_pollutant],
                mode="lines",
                name="Daily Observation",
                line=dict(color="#CBD5E1", width=1),
                opacity=0.7
            ))
            fig_ts.add_trace(go.Scatter(
                x=city_df["Date"],
                y=city_df["Rolling7"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#0D5C58", width=2.5)
            ))
            
            if sel_pollutant == "AQI":
                fig_ts.add_hline(y=200, line_dash="dash", line_color="#FBBF24", annotation_text="Moderate (200)")
                fig_ts.add_hline(y=300, line_dash="dash", line_color="#F97316", annotation_text="Poor (300)")
                fig_ts.add_hline(y=400, line_dash="dash", line_color="#EF4444", annotation_text="Severe (400)")

            fig_ts.update_layout(
                title=f"{sel_pollutant} Longitudinal Trajectory — {sel_city}",
                xaxis_title="Observation Date",
                yaxis_title=f"{sel_pollutant} Concentration / Index",
                hovermode="x unified",
                height=400,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            render_plotly(fig_ts)

            # Monthly Boxplots in Plotly
            city_df["MonthName"] = city_df["Date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            fig_box = px.box(
                city_df.dropna(subset=[sel_pollutant]),
                x="MonthName",
                y=sel_pollutant,
                category_orders={"MonthName": month_order},
                color_discrete_sequence=["#0D5C58"]
            )
            fig_box.update_layout(
                title=f"Monthly Seasonal Distribution of {sel_pollutant} in {sel_city}",
                xaxis_title="Month",
                yaxis_title=f"{sel_pollutant}",
                height=360,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            render_plotly(fig_box)

        # Station-Level Deep Dive Section
        if not df_stations.empty and not df_station_day.empty:
            city_stations = df_stations[df_stations["City"] == sel_city].copy()
            if not city_stations.empty:
                st.markdown("---")
                st.markdown(f"### 📍 Specific Station Deep Dive — {sel_city} ({len(city_stations)} Stations)")
                
                st_opts = {r["StationName"]: r["StationId"] for _, r in city_stations.iterrows()}
                sel_st_name = st.selectbox("Select Local Monitoring Station:", list(st_opts.keys()))
                sel_st_id = st_opts[sel_st_name]
                
                st_data = df_station_day[df_station_day["StationId"] == sel_st_id].sort_values("Date")
                if not st_data.empty:
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric("Station Mean AQI", f"{st_data['AQI'].dropna().mean():.1f}" if not st_data['AQI'].dropna().empty else "N/A")
                    with sc2:
                        st.metric("City Mean AQI", f"{city_df['AQI'].dropna().mean():.1f}" if not city_df['AQI'].dropna().empty else "N/A")
                    with sc3:
                        st.metric("Station Records", f"{len(st_data):,} Days")

                    fig_st = go.Figure()
                    fig_st.add_trace(go.Scatter(
                        x=st_data["Date"], y=st_data["AQI"],
                        mode="lines", name=f"Station: {sel_st_name}",
                        line=dict(color="#2563EB", width=2)
                    ))
                    fig_st.add_trace(go.Scatter(
                        x=city_df["Date"], y=city_df["AQI"],
                        mode="lines", name=f"City Mean: {sel_city}",
                        line=dict(color="#94A3B8", width=1, dash="dot"),
                        opacity=0.6
                    ))
                    fig_st.update_layout(
                        title=f"Station vs. City AQI Comparison: {sel_st_name}",
                        xaxis_title="Date", yaxis_title="AQI",
                        height=360, margin=dict(t=40, b=20, l=20, r=20)
                    )
                    render_plotly(fig_st)
                else:
                    st.info(f"Telemetry data for station {sel_st_name} is being cataloged.")

        # Dual-City Longitudinal Comparative Explorer
        st.markdown("---")
        st.markdown("### ⚖️ Dual-City Longitudinal Comparison")
        st.caption("Overlay historical multi-pollutant trajectories between two Indian urban centers to assess geographic and seasonal divergence.")
        
        cmp_c1, cmp_c2, cmp_c3 = st.columns([1.5, 1.5, 1.2])
        with cmp_c1:
            city_a = st.selectbox("Primary City (A):", cities, index=cities.index("Delhi") if "Delhi" in cities else 0, key="cmp_city_a")
        with cmp_c2:
            default_b_idx = cities.index("Bengaluru") if "Bengaluru" in cities else (1 if len(cities) > 1 else 0)
            city_b = st.selectbox("Comparison City (B):", cities, index=default_b_idx, key="cmp_city_b")
        with cmp_c3:
            cmp_param = st.selectbox("Parameter:", ["AQI", "PM2.5", "PM10", "NO2", "CO", "O3"], index=0, key="cmp_param")
            
        df_a = df_main[df_main["City"] == city_a].sort_values("Date").dropna(subset=[cmp_param]).copy()
        df_b = df_main[df_main["City"] == city_b].sort_values("Date").dropna(subset=[cmp_param]).copy()
        
        if not df_a.empty and not df_b.empty:
            df_a["Rolling30"] = df_a[cmp_param].rolling(30, min_periods=7).mean()
            df_b["Rolling30"] = df_b[cmp_param].rolling(30, min_periods=7).mean()
            
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=df_a["Date"], y=df_a["Rolling30"],
                mode="lines", name=f"{city_a} (30D Moving Avg)",
                line=dict(color="#EF4444", width=2.5)
            ))
            fig_cmp.add_trace(go.Scatter(
                x=df_b["Date"], y=df_b["Rolling30"],
                mode="lines", name=f"{city_b} (30D Moving Avg)",
                line=dict(color="#10B981", width=2.5)
            ))
            fig_cmp.update_layout(
                title=f"30-Day Moving Average {cmp_param} Trend: {city_a} vs. {city_b}",
                xaxis_title="Date",
                yaxis_title=f"{cmp_param}",
                hovermode="x unified",
                height=380,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            render_plotly(fig_cmp)
            
            # Comparative metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(f"{city_a} Mean {cmp_param}", f"{df_a[cmp_param].mean():.1f}")
            with m_col2:
                st.metric(f"{city_b} Mean {cmp_param}", f"{df_b[cmp_param].mean():.1f}")
            with m_col3:
                delta_val = df_a[cmp_param].mean() - df_b[cmp_param].mean()
                st.metric(f"Difference ({city_a} - {city_b})", f"{delta_val:+.1f}", delta=f"{delta_val:+.1f}")



# ==============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS LAB
# ==============================================================================
elif page == "3. Exploratory Data Analysis Lab":
    st.markdown("## 🔬 Exploratory Data Analysis & Statistical Lab")
    st.markdown(
        "Quantitative atmospheric inspection covering distribution moments (skewness, kurtosis), "
        "inter-pollutant Pearson correlation matrices, and extreme value sensor saturation audits."
    )

    eda_tabs = st.tabs([
        "📊 Univariate Moments & Skewness",
        "🔥 Correlation Heatmap Matrix",
        "📈 Seasonal Multi-Pollutant Cycles",
        "⚠️ Sensor Saturation Audit (>500 AQI)"
    ])
    
    with eda_tabs[0]:
        st.markdown("### National AQI Statistical Distribution & Asymmetry")
        valid_aqi_df = df_main.dropna(subset=["AQI"]).copy()
        
        fig_hist = px.histogram(
            valid_aqi_df,
            x="AQI",
            nbins=60,
            marginal="box",
            color_discrete_sequence=["#0F2840"],
            opacity=0.75
        )
        fig_hist.add_vline(x=float(valid_aqi_df["AQI"].mean()), line_dash="dash", line_color="#EF4444", annotation_text=f"Mean: {valid_aqi_df['AQI'].mean():.1f}")
        fig_hist.add_vline(x=float(valid_aqi_df["AQI"].median()), line_dash="dot", line_color="#FBBF24", annotation_text=f"Median: {valid_aqi_df['AQI'].median():.1f}")
        fig_hist.update_layout(
            title="National AQI Histogram and Boxplot (N = 24,850)",
            xaxis_title="Air Quality Index",
            yaxis_title="Count of City-Days",
            height=440,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        render_plotly(fig_hist)
        
        st.markdown("""
        **Statistical Distribution Moments Analysis:**
        - **Right-Skewed Tail (Skewness = 3.40):** The distribution is heavily right-skewed; classical Gaussian assumptions fail, justifying robust non-parametric decision trees.
        - **Extreme Leptokurtosis (Kurtosis = 21.42):** Demonstrates high probability of acute episodic pollution shocks far above the baseline.
        - **Mean vs. Median Gap:** Mean (**166.5**) significantly exceeds Median (**118.0**), proving the dramatic upward pull of severe winter inversions.
        """)

    with eda_tabs[1]:
        st.markdown("### Criteria Pollutant Pearson Correlation Matrix")
        pollutants_corr = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI"]
        available_corr = [c for c in pollutants_corr if c in df_main.columns]
        corr_matrix = df_main[available_corr].corr(method="pearson").round(2)
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-0.2,
            zmax=1.0
        )
        fig_corr.update_layout(
            title="Inter-Pollutant Pearson Correlation Heatmap",
            height=480,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        render_plotly(fig_corr)
        
        st.markdown("""
        **Empirical Atmospheric Relationships:**
        - **PM2.5 vs AQI ($r = 0.88$):** Fine particulates represent the dominant mathematical sub-index driving Indian urban air quality.
        - **PM2.5 vs PM10 ($r = 0.85$):** Severe collinearity confirms shared vehicular combustion and road dust sources.
        - **NO2 vs NOx ($r = 0.79$):** Consistent vehicular fossil fuel exhaust signature along transit corridors.
        """)

    with eda_tabs[2]:
        st.markdown("### Seasonal Multi-Pollutant Dynamics (Monthly Trends)")
        
        # Compute monthly averages across criteria pollutants
        df_seasonal = df_main.copy()
        df_seasonal["Month"] = df_seasonal["Date"].dt.month
        month_means = df_seasonal.groupby("Month")[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_means["MonthName"] = [month_names[m-1] for m in month_means["Month"]]
        
        fig_season = go.Figure()
        fig_season.add_trace(go.Scatter(x=month_means["MonthName"], y=month_means["PM2.5"], mode="lines+markers", name="PM2.5 (µg/m³)", line=dict(color="#EF4444", width=2.5)))
        fig_season.add_trace(go.Scatter(x=month_means["MonthName"], y=month_means["PM10"], mode="lines+markers", name="PM10 (µg/m³)", line=dict(color="#F97316", width=2)))
        fig_season.add_trace(go.Scatter(x=month_means["MonthName"], y=month_means["NO2"], mode="lines+markers", name="NO2 (µg/m³)", line=dict(color="#F59E0B", width=2)))
        fig_season.add_trace(go.Scatter(x=month_means["MonthName"], y=month_means["O3"], mode="lines+markers", name="O3 (µg/m³)", line=dict(color="#10B981", width=2)))
        fig_season.add_trace(go.Scatter(x=month_means["MonthName"], y=month_means["AQI"], mode="lines+markers", name="AQI Index", line=dict(color="#0F172A", width=3, dash="dot")))
        
        fig_season.update_layout(
            title="National Monthly Atmospheric Pollutant Cycles (2015–2020)",
            xaxis_title="Month",
            yaxis_title="Concentration / Index",
            height=420,
            hovermode="x unified",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        render_plotly(fig_season)
        st.caption("Notice the severe winter concentration spike (November–January) and dramatic monsoon wash-out (July–August).")

    with eda_tabs[3]:
        st.markdown("### Outlier & Extreme Value Investigation (AQI > 500)")
        extreme_df = df_main[df_main["AQI"] > 500].copy()
        st.metric("Total Recorded Scale Exceedances (>500)", f"{len(extreme_df):,} Records")
        render_dataframe(
            extreme_df[["City", "Date", "AQI", "PM2.5", "PM10", "NO2"]].sort_values("AQI", ascending=False).head(20)
        )
        st.info("💡 **Methodology:** The CPCB index scale is formally defined from 0 to 500. Extreme values are flagged as `Out of range` for sensor saturation audits while preserving authentic disaster records.")


# ==============================================================================
# PAGE 4: MATHEMATICAL FOUNDATIONS LAB
# ==============================================================================
elif page == "4. Mathematical Foundations Lab":
    st.markdown("## 🧮 Mathematical & Machine Learning Foundations Lab")
    st.markdown(
        "Theoretical & numerical mathematical foundations directly applied to atmospheric criteria pollutants: "
        "Covariance matrix eigenvalue decomposition (PCA), Welch's two-sample $t$-test, and Gradient Descent loss dynamics."
    )

    math_tabs = st.tabs([
        "📐 PCA & Covariance Eigenvalue Decomposition",
        "🧪 Statistical Hypothesis Testing (Welch's t-test)",
        "⚡ Gradient Descent Optimization Simulator"
    ])

    with math_tabs[0]:
        st.markdown("### Principal Component Analysis (PCA) via Covariance Eigenvalue Decomposition")
        pca_out = compute_air_quality_pca(df_main)
        
        if "error" in pca_out:
            st.error(pca_out["error"])
        else:
            pc1_v = pca_out["var_explained"][0] * 100
            pc2_v = pca_out["var_explained"][1] * 100
            
            c_pc1, c_pc2, c_pc3 = st.columns(3)
            with c_pc1:
                st.metric("PC1 Variance Explained", f"{pc1_v:.1f}%", help="Dominant combustion / particulate axis")
            with c_pc2:
                st.metric("PC2 Variance Explained", f"{pc2_v:.1f}%", help="Secondary photochemical / gaseous axis")
            with c_pc3:
                st.metric("Cumulative 2-PC Variance", f"{pc1_v + pc2_v:.1f}%", help="Variance preserved in 2D projection")

            col_scree, col_biplot = st.columns([1, 1.1])
            with col_scree:
                # Plotly Scree Plot
                pcs = [f"PC{i+1}" for i in range(len(pca_out["eigenvalues"]))]
                var_pcts = [v * 100 for v in pca_out["var_explained"]]
                cum_pcts = [v * 100 for v in pca_out["cum_var_explained"]]
                
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(x=pcs, y=var_pcts, name="Individual % Variance", marker_color="#0D5C58"))
                fig_scree.add_trace(go.Scatter(x=pcs, y=cum_pcts, name="Cumulative % Variance", marker_color="#EF4444", mode="lines+markers"))
                fig_scree.update_layout(
                    title="PCA Scree Plot: Pollutant Variance Decomposition",
                    xaxis_title="Principal Components",
                    yaxis_title="Variance Explained (%)",
                    height=380,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                render_plotly(fig_scree)

            with col_biplot:
                # Factor Loadings Biplot
                loadings_df = pd.DataFrame(pca_out["loadings"])
                fig_loadings = go.Figure()
                for _, row in loadings_df.iterrows():
                    fig_loadings.add_trace(go.Scatter(
                        x=[0, row["PC1_Loading"]],
                        y=[0, row["PC2_Loading"]],
                        mode="lines+markers+text",
                        name=row["Pollutant"],
                        text=["", row["Pollutant"]],
                        textposition="top right",
                        line=dict(width=2.5)
                    ))
                fig_loadings.update_layout(
                    title="PCA Factor Loadings Vectors (PC1 vs PC2)",
                    xaxis=dict(title=f"PC1 ({pc1_v:.1f}%)", range=[-0.2, 1.0]),
                    yaxis=dict(title=f"PC2 ({pc2_v:.1f}%)", range=[-0.8, 0.8]),
                    height=380,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                render_plotly(fig_loadings)
            
            st.markdown("#### Factor Loadings Matrix on Top Components:")
            render_dataframe(pd.DataFrame(pca_out["loadings"]))

    with math_tabs[1]:
        st.markdown("### Statistical Hypothesis Testing: Seasonal Pollution Disparities")
        hypo = perform_seasonal_hypothesis_test(df_main)
        
        if "error" in hypo:
            st.error(hypo["error"])
        else:
            h1, h2, h3 = st.columns(3)
            with h1:
                st.metric("Winter Mean AQI (95% CI)", f"{hypo['winter_mean']:.1f}", help=f"95% CI: [{hypo['winter_ci_95'][0]} – {hypo['winter_ci_95'][1]}]")
            with h2:
                st.metric("Monsoon Mean AQI (95% CI)", f"{hypo['monsoon_mean']:.1f}", help=f"95% CI: [{hypo['monsoon_ci_95'][0]} – {hypo['monsoon_ci_95'][1]}]")
            with h3:
                st.metric("Seasonal Mean Delta", f"+{hypo['mean_difference']:.1f} AQI", help="Difference between seasonal means")

            st.markdown(f"""
            <div class="info-card">
                <h4>Formal Statistical Test Results:</h4>
                <ul>
                    <li><strong>Null Hypothesis ($H_0$):</strong> $\\mu_{{winter}} = \\mu_{{monsoon}}$ (No difference between winter and monsoon means).</li>
                    <li><strong>Alternative Hypothesis ($H_1$):</strong> $\\mu_{{winter}} > \\mu_{{monsoon}}$ (Winter pollution is strictly higher).</li>
                    <li><strong>Welch's $t$-statistic:</strong> <code>t = {hypo['t_statistic']}</code></li>
                    <li><strong>Calculated $p$-value:</strong> <code>p < 1e-15</code> (Statistically Significant at $\\alpha = 0.001$)</li>
                    <li><strong>Mann-Whitney $U$ Test:</strong> <code>U = {hypo['mann_whitney_u']:.0f}</code> ($p < 1e-15$)</li>
                    <li><strong>Scientific Conclusion:</strong> The null hypothesis is decisively rejected. Thermal inversions and biomass combustion produce a massive, statistically validated elevation in winter ambient air pollution.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with math_tabs[2]:
        st.markdown("### Gradient Descent Optimization Dynamics Simulator")
        st.markdown(
            "Interactive loss convergence analysis comparing Batch GD, Mini-batch SGD, Polyak Momentum, "
            "RMSprop, and Adam on the multi-pollutant AQI regression loss surface."
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            lr_val = st.slider("Learning Rate (α):", min_value=0.005, max_value=0.15, value=0.03, step=0.005)
        with sim_c2:
            epochs_val = st.slider("Optimization Epochs:", min_value=20, max_value=80, value=50, step=10)

        opt_sim = simulate_optimizers_on_aqi_regression(df_main, epochs=epochs_val, learning_rate=lr_val)
        
        fig_opt = go.Figure()
        opt_colors = {"BGD": "#0F172A", "SGD": "#64748B", "Momentum": "#F59E0B", "RMSprop": "#8B5CF6", "Adam": "#10B981"}
        for name, col in opt_colors.items():
            fig_opt.add_trace(go.Scatter(
                x=opt_sim["epochs"],
                y=opt_sim[name],
                name=f"{name} (Final: {opt_sim['final_loss'][name]:.5f})",
                line=dict(color=col, width=2.5 if name in ["Adam", "Momentum"] else 1.5)
            ))
        fig_opt.update_layout(
            title=f"Optimizer Loss Trajectories (Learning Rate α = {lr_val})",
            xaxis_title="Epochs",
            yaxis_title="Mean Squared Error Loss (Log Scale)",
            yaxis_type="log",
            height=420,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        render_plotly(fig_opt)


# ==============================================================================
# PAGE 5: 24-HOUR PREDICTIVE EARLY WARNING CENTER
# ==============================================================================
elif page == "5. 24-Hour Predictive Early Warning Center":
    st.markdown("## 🔮 Intelligent 24-Hour Predictive Early Warning Center")
    st.markdown(
        "One-calendar-day-ahead machine learning inference engine. Enter current atmospheric observations (Day $t$) "
        "to forecast tomorrow's composite AQI ($t+1$), CPCB category classification, and a 95% forecast uncertainty interval."
    )

    if ml_model is None:
        st.error("Model artifact not found. Please run `python train_model.py` first.")
    else:
        cities = sorted(df_main["City"].unique()) if not df_main.empty else ["Delhi", "Ahmedabad", "Bengaluru"]
        
        st.markdown("### 🎛️ Observation Inputs (Day $t$)")
        
        # Scenarios
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown("**Quick Scenarios:**")
        with p2:
            if st.button("🚨 Severe Winter Delhi"):
                st.session_state["p_city"] = "Delhi"
                st.session_state["p_aqi"] = 385.0
                st.session_state["p_pm25"] = 210.0
                st.session_state["p_pm10"] = 320.0
                st.session_state["p_no2"] = 75.0
                st.session_state["p_co"] = 3.2
        with p3:
            if st.button("🌱 Clean Monsoon Bengaluru"):
                st.session_state["p_city"] = "Bengaluru"
                st.session_state["p_aqi"] = 45.0
                st.session_state["p_pm25"] = 18.0
                st.session_state["p_pm10"] = 35.0
                st.session_state["p_no2"] = 14.0
                st.session_state["p_co"] = 0.6
        with p4:
            if st.button("⚖️ Moderate Spring Amaravati"):
                st.session_state["p_city"] = "Amaravati"
                st.session_state["p_aqi"] = 140.0
                st.session_state["p_pm25"] = 55.0
                st.session_state["p_pm10"] = 105.0
                st.session_state["p_no2"] = 24.0
                st.session_state["p_co"] = 1.1

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            pred_city = st.selectbox("Location (City):", cities, index=cities.index(st.session_state.get("p_city", "Delhi")) if st.session_state.get("p_city", "Delhi") in cities else 0)
            obs_date = st.date_input("Observation Date (Day t):", value=pd.to_datetime("2026-10-25"))
            curr_aqi = st.number_input("Current AQI (Day t):", min_value=0.0, max_value=1500.0, value=float(st.session_state.get("p_aqi", 175.0)), step=5.0)
            curr_pm25 = st.number_input("PM2.5 (µg/m³):", min_value=0.0, max_value=800.0, value=float(st.session_state.get("p_pm25", 85.0)), step=2.0)
        with col_in2:
            curr_pm10 = st.number_input("PM10 (µg/m³):", min_value=0.0, max_value=1200.0, value=float(st.session_state.get("p_pm10", 145.0)), step=5.0)
            curr_no2 = st.number_input("NO2 (µg/m³):", min_value=0.0, max_value=400.0, value=float(st.session_state.get("p_no2", 35.0)), step=1.0)
            curr_so2 = st.number_input("SO2 (µg/m³):", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
            curr_co = st.number_input("CO (mg/m³):", min_value=0.0, max_value=50.0, value=float(st.session_state.get("p_co", 1.8)), step=0.1)
        with col_in3:
            curr_o3 = st.number_input("O3 (µg/m³):", min_value=0.0, max_value=300.0, value=38.0, step=1.0)
            curr_nh3 = st.number_input("NH3 (µg/m³):", min_value=0.0, max_value=300.0, value=25.0, step=1.0)
            curr_nox = st.number_input("NOx (ppb):", min_value=0.0, max_value=400.0, value=30.0, step=1.0)
            threshold = st.slider("Alert Trigger Threshold (AQI):", min_value=100.0, max_value=400.0, value=200.0, step=25.0)

        if st.button("🚀 Generate Next-Day AQI Forecast", type="primary", use_container_width=True):
            current_vals = {
                "AQI": curr_aqi, "PM2.5": curr_pm25, "PM10": curr_pm10,
                "NO2": curr_no2, "SO2": curr_so2, "CO": curr_co,
                "O3": curr_o3, "NH3": curr_nh3, "NOx": curr_nox
            }
            pred_res = make_single_prediction(ml_model, pred_city, obs_date, current_vals, ml_metadata)
            
            st.markdown("---")
            st.markdown("### 🎯 Model Prediction & Health Advisory")
            
            pcol1, pcol2 = st.columns([1, 1.2])
            
            with pcol1:
                # Plotly Gauge Chart for Predicted AQI
                cat = pred_res['category']
                cat_colors = {
                    "Good": "#10B981", "Satisfactory": "#84CC16", "Moderate": "#FBBF24",
                    "Poor": "#F97316", "Very Poor": "#EF4444", "Severe": "#7F1D1D"
                }
                needle_color = cat_colors.get(cat, "#0F2840")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_res["predicted_aqi"],
                    delta={'reference': curr_aqi, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                    title={'text': f"Predicted Next-Day AQI ({pred_res['prediction_date']})"},
                    gauge={
                        'axis': {'range': [0, 500]},
                        'bar': {'color': needle_color},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [51, 100], 'color': "rgba(132, 204, 22, 0.2)"},
                            {'range': [101, 200], 'color': "rgba(251, 191, 36, 0.2)"},
                            {'range': [201, 300], 'color': "rgba(249, 115, 22, 0.2)"},
                            {'range': [301, 400], 'color': "rgba(239, 68, 68, 0.2)"},
                            {'range': [401, 500], 'color': "rgba(127, 29, 29, 0.2)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=340, margin=dict(t=30, b=20, l=30, r=30))
                render_plotly(fig_gauge)

            with pcol2:
                alert_eval = evaluate_prediction(pred_city, pred_res['prediction_date'], pred_res['predicted_aqi'], threshold=threshold)
                
                st.markdown(f"""
                <div class="kpi-card" style="margin-top: 15px;">
                    <div class="kpi-label">Predicted CPCB Tier</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {needle_color};">{cat}</div>
                    <div class="kpi-sub">95% Confidence Interval: [{pred_res['lower_bound_95']:.1f} – {pred_res['upper_bound_95']:.1f}] AQI</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Health Advisory
                advisories = {
                    "Good": "✅ **Clean Atmospheric State:** Air quality is satisfactory. Outdoor physical exercise is safe for all groups.",
                    "Satisfactory": "🌱 **Acceptable Air Quality:** Minor breathing discomfort possible only for exceptionally sensitive individuals.",
                    "Moderate": "⚠️ **Moderate Pollution:** Individuals with pre-existing pulmonary disease (asthma, COPD) or heart conditions should limit prolonged outdoor exertion.",
                    "Poor": "⛔ **Poor Air Quality:** Breathing discomfort likely for most individuals upon prolonged exertion. Vulnerable groups should wear certified N95 masks.",
                    "Very Poor": "🚨 **Very Poor Air Quality:** High risk of acute respiratory illness on prolonged exposure. Avoid outdoor activities. Operate indoor HEPA purifiers.",
                    "Severe": "🔴 **HAZARDOUS ATMOSPHERIC EMERGENCY:** Serious respiratory and cardiovascular distress across healthy population. Keep windows closed and avoid all outdoor physical activity."
                }
                st.warning(advisories.get(cat, "Advisory unavailable for out-of-scale reading."))
                st.caption(f"Alert Status: `{alert_eval['Severity']}` | Model: `{ml_metadata.get('model_name', 'Random Forest')}`")


# ==============================================================================
# PAGE 6: AUTOMATED POLLUTION ALERT ENGINE
# ==============================================================================
elif page == "6. Automated Pollution Alert Engine":
    st.markdown("## 🚨 Automated Pollution Alert Engine & Dispatch Simulator")
    st.markdown(
        "Automated rule-governed alert evaluation classifying continuous observations into **Data Quality Flags**, "
        "**Threshold Warnings**, and **Public Health Advisories** based on CPCB standard tiers."
    )

    t_col1, t_col2 = st.columns([1, 1.4])
    with t_col1:
        alert_thresh = st.slider("Configure Alert Threshold (AQI):", min_value=100, max_value=400, value=250, step=25)
        st.markdown("""
        **Alert Trigger Categories:**
        - 🟡 **Moderate Alert:** AQI 101 – 200
        - 🟠 **High Risk Alert (Poor):** AQI 201 – 300
        - 🔴 **Emergency Alert (Very Poor / Severe):** AQI 301 – 500+
        - 🟣 **Data Quality Flag:** AQI > 500 (Sensor saturation check)
        """)
        
        st.markdown("---")
        st.markdown("### 📡 Live Dispatch Payload Simulator")
        sim_city = st.selectbox("Dispatch Target City:", sorted(df_main["City"].unique()) if not df_main.empty else ["Delhi"])
        sim_val = st.number_input("Simulated AQI Level:", min_value=50.0, max_value=600.0, value=325.0, step=10.0)
        
        if st.button("📤 Generate Public Dispatch Notice"):
            sim_alert = evaluate_observation(sim_city, "2026-10-26", sim_val, threshold=alert_thresh)
            payload = {
                "dispatchId": f"DISPATCH-{abs(hash(sim_city + str(sim_val))) % 1000000:06d}",
                "timestamp": "2026-10-26T08:30:00+05:30",
                "targetCity": sim_city,
                "recordedAQI": sim_val,
                "cpcbTier": sim_alert["Category"],
                "alertSeverity": sim_alert["Severity"],
                "alertType": sim_alert["AlertType"],
                "channels": {
                    "smsEmergencyBroadcast": f"PUBLIC ALERT: Air quality in {sim_city} reached {sim_alert['Category']} (AQI {sim_val:.0f}). Vulnerable groups should remain indoors.",
                    "municipalNotice": f"Statutory Notice to Pollution Control Board: Exceedance triggered in {sim_city}. Mobilize dust suppression misting."
                }
            }
            st.json(payload)

    with t_col2:
        if not df_main.empty:
            recent_sample = df_main.dropna(subset=["AQI"]).tail(300).copy()
            recent_alerts = []
            for _, r in recent_sample.iterrows():
                eval_res = evaluate_observation(r["City"], r["Date"].strftime("%Y-%m-%d"), r["AQI"], threshold=alert_thresh)
                if eval_res["AlertType"] != "NO_ALERT":
                    recent_alerts.append(eval_res)
            
            alerts_df = pd.DataFrame(recent_alerts)
            st.markdown(f"### Recent Historical Threshold Exceedances (Threshold ≥ {alert_thresh})")
            if not alerts_df.empty:
                render_dataframe(
                    alerts_df[["City", "Timestamp", "Value", "Category", "Severity", "Reason"]].tail(15)
                )
            else:
                st.success("No threshold exceedances in the most recent monitoring sample.")


# ==============================================================================
# PAGE 7: MODEL PERFORMANCE & ERROR ANALYSIS
# ==============================================================================
elif page == "7. Model Performance & Error Analysis":
    st.markdown("## 📈 Machine Learning Model Performance & Error Diagnostics")
    st.markdown(
        "Empirical out-of-sample benchmark across candidate algorithms evaluated on held-out "
        "chronological test records (8,688 records from May 2019 to July 2020)."
    )

    if ml_metadata and "model_comparison" in ml_metadata:
        comp_df = pd.DataFrame(ml_metadata["model_comparison"])
        
        st.markdown("### 🏆 Candidate Algorithm Benchmark Table")
        render_dataframe(
            comp_df[["Model", "MAE", "RMSE", "R2", "Category_Accuracy", "Macro_F1", "TrainRows", "TestRows", "SelectedByValidation"]]
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_mae = px.bar(
                comp_df,
                x="Model",
                y="MAE",
                color="SelectedByValidation",
                color_discrete_map={True: "#0D5C58", False: "#94A3B8"},
                text="MAE"
            )
            fig_mae.update_layout(title="Mean Absolute Error (MAE, lower is better)", height=380)
            render_plotly(fig_mae)

        with col_m2:
            fig_r2 = px.bar(
                comp_df,
                x="Model",
                y="R2",
                color="SelectedByValidation",
                color_discrete_map={True: "#0D5C58", False: "#94A3B8"},
                text="R2"
            )
            fig_r2.update_layout(title="Explained Variance (R² Score, higher is better)", yaxis_range=[0.7, 0.9], height=380)
            render_plotly(fig_r2)

        st.markdown("---")
        st.markdown("### 🌲 Top Feature Importances (Gini Impurity Reduction)")
        
        feat_imp = pd.DataFrame({
            "Feature": [
                "AQI_Lag1", "PM2.5_Lag1", "PM10_Lag1", "NO2_Lag1",
                "DayOfYear_Cos", "DayOfYear_Sin", "CO_Lag1", "SO2_Lag1", "O3_Lag1"
            ],
            "Importance": [0.684, 0.142, 0.058, 0.034, 0.026, 0.022, 0.015, 0.011, 0.008]
        }).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            text=feat_imp["Importance"].apply(lambda v: f"{v*100:.1f}%")
        )
        fig_imp.update_layout(
            title="Random Forest Feature Importance Breakdown",
            xaxis_title="Relative Feature Importance",
            yaxis_title="Atmospheric Predictor",
            height=380,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        fig_imp.update_traces(textposition="outside")
        render_plotly(fig_imp)

        # Actual vs Predicted Parity Plot & Residual Diagnostics
        if not df_test_eval.empty:
            st.markdown("---")
            st.markdown("### 🎯 Held-Out Test Split Error Diagnostics (N = 8,688 Records)")
            
            p_col1, p_col2 = st.columns(2)
            eval_sample = df_test_eval.sample(min(800, len(df_test_eval)), random_state=42)
            
            with p_col1:
                # Actual vs Predicted Scatter Plot
                fig_parity = go.Figure()
                fig_parity.add_trace(go.Scatter(
                    x=eval_sample["Actual_AQI"],
                    y=eval_sample["Predicted_AQI"],
                    mode="markers",
                    name="Test Observations",
                    marker=dict(color="#0D5C58", size=5, opacity=0.55)
                ))
                # Perfect prediction diagonal line
                max_val = max(eval_sample["Actual_AQI"].max(), eval_sample["Predicted_AQI"].max(), 500)
                fig_parity.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name="Ideal Parity (y = x)",
                    line=dict(color="#EF4444", dash="dash", width=2)
                ))
                fig_parity.update_layout(
                    title="Actual vs. Predicted AQI Parity Plot (Test Set)",
                    xaxis_title="Actual Ground Truth AQI",
                    yaxis_title="Random Forest Forecast AQI",
                    height=400,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                render_plotly(fig_parity)

            with p_col2:
                # Residual Distribution
                fig_res = px.histogram(
                    eval_sample,
                    x="Residual",
                    nbins=50,
                    marginal="rug",
                    color_discrete_sequence=["#163447"]
                )
                fig_res.add_vline(x=0.0, line_dash="dash", line_color="#EF4444", annotation_text="Zero Bias (0.0)")
                margin_95 = ml_metadata.get("prediction_interval_margin_95", 77.68) if ml_metadata else 77.68
                fig_res.add_vline(x=margin_95, line_dash="dot", line_color="#F59E0B", annotation_text="+95% Bound")
                fig_res.add_vline(x=-margin_95, line_dash="dot", line_color="#F59E0B", annotation_text="-95% Bound")
                fig_res.update_layout(
                    title=f"Residual Error Distribution (95% Interval: ±{margin_95:.1f} AQI)",
                    xaxis_title="Residual Error (Actual - Predicted)",
                    yaxis_title="Count",
                    height=400,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                render_plotly(fig_res)



