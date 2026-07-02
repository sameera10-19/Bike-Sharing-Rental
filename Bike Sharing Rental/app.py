import os
import json
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Model Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Loads the pre-trained best model and scaling matrix saved during training."""
    model_path = os.path.join(APP_DIR, "best_model.pkl")
    scaler_path = os.path.join(APP_DIR, "scaler.pkl")
    
    m = joblib.load(model_path)
    s = joblib.load(scaler_path)
    return m, s

@st.cache_data
def load_evaluation_metrics():
    """Loads validation metrics compiled across evaluated algorithms."""
    path = os.path.join(APP_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return sorted(
            [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
             for k, v in raw.items()],
            key=lambda r: r["RMSE"],
        )
    
    # Fallback to recorded training outputs if JSON is missing
    return [
        {"Model": "GB (Tuned)",        "MAE": 21.70, "RMSE": 35.69, "R²": 0.9598},
        {"Model": "Gradient Boosting", "MAE": 25.06, "RMSE": 39.68, "R²": 0.9503},
        {"Model": "Random Forest",     "MAE": 24.22, "RMSE": 40.98, "R²": 0.9470},
        {"Model": "DT (Tuned)",        "MAE": 31.66, "RMSE": 54.35, "R²": 0.9067},
        {"Model": "RF (Tuned)",        "MAE": 38.01, "RMSE": 58.13, "R²": 0.8933},
        {"Model": "Decision Tree",     "MAE": 34.08, "RMSE": 59.03, "R²": 0.8900},
    ]

try:
    model, scaler = load_model_artifacts()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Application Header ────────────────────────────────────────────────────────
st.title("🚲 Bike Rental Demand Predictor")
st.markdown(
    "Predict hourly deployment demand for municipal bike sharing systems "
    "using an optimized Gradient Boosting regressor model engine."
)

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Feature Space Controls")

    # Temporal Fields
    st.subheader("📅 Temporal Settings")
    selected_date = st.date_input(
        "Evaluation Date",
        value=datetime.date(2012, 6, 15),
        min_value=datetime.date(2011, 1, 1),
        max_value=datetime.date(2012, 12, 31),
        help="Model baseline configured around a 2-year operational envelope.",
    )
    hour = st.slider("Hour Window (24h)", 0, 23, 12, format="%d:00")

    # Environmental Categoricals
    st.subheader("🌤 Seasonal & Environmental")
    season = st.selectbox("Current Season", ["Fall", "Spring", "Summer", "Winter"])
    weather = st.selectbox(
        "Atmospheric Condition",
        ["Clear / Few Clouds", "Mist / Cloudy", "Light Snow / Rain", "Heavy Rain / Thunderstorm"],
    )

    # Continuous Environmental Metrics
    st.subheader("🌡 Weather Scale Vectors")
    temp_c   = st.slider("Real Temperature (°C)", -8.0, 41.0, 20.0, 0.5)
    atemp_c  = st.slider("Feels-Like Index (°C)",  -16.0, 50.0, 18.0, 0.5)
    humidity = st.slider("Relative Humidity (%)",    0,  100,  60)
    wind_kmh = st.slider("Wind Speed Vector (km/h)", 0.0, 57.0, 15.0, 0.5)

    # Operational Status Flags
    st.subheader("📆 Calendar Constraints")
    workingday = st.checkbox("Standard Working Day", value=True)
    holiday    = st.checkbox("Public Statutory Holiday", value=False)

# ── Pipeline Vector Generation & Feature Engineering ─────────────────────────
yr          = float(selected_date.year)
mnth        = float(selected_date.month)
py_wd       = selected_date.weekday()       
weekday_ds  = float((py_wd + 1) % 7)       # Scale mapping conversion: Sun=0 to Sat=6
day_of_year = float(selected_date.timetuple().tm_yday)
is_weekend  = 1.0 if int(weekday_ds) in (0, 6) else 0.0

# Structural MinMax Clipping Maps
temp_norm  = float(np.clip(temp_c   / 41.0,  0, 1))
atemp_norm = float(np.clip(atemp_c  / 50.0,  0, 1))
hum_norm   = float(np.clip(humidity / 100.0, 0, 1))
wind_norm  = float(np.clip(wind_kmh / 57.0,  0, 1))

# One-Hot Matrix Transformation Encodings
season_springer = 1.0 if season == "Spring" else 0.0
season_summer   = 1.0 if season == "Summer" else 0.0
season_winter   = 1.0 if season == "Winter" else 0.0

ws_heavy = 1.0 if weather == "Heavy Rain / Thunderstorm" else 0.0
ws_snow  = 1.0 if weather == "Light Snow / Rain"          else 0.0
ws_mist  = 1.0 if weather == "Mist / Cloudy"              else 0.0

# Conditional Flag Assertions
wd_flag  = 1.0 if (workingday and not holiday) else 0.0
hol_flag = 1.0 if holiday else 0.0

# Immutable Global Tracking Sequence Alignment Map
FEATURE_COLS = [
    "yr", "mnth", "hr", "weekday", "temp", "atemp", "hum", "windspeed",
    "day_of_year", "is_weekend",
    "season_springer", "season_summer", "season_winter",
    "weathersit_Heavy Rain", "weathersit_Light Snow", "weathersit_Mist",
    "workingday_Working Day", "holiday_Yes",
]

input_df = pd.DataFrame([{
    "yr": yr, "mnth": mnth, "hr": float(hour), "weekday": weekday_ds,
    "temp": temp_norm, "atemp": atemp_norm, "hum": hum_norm, "windspeed": wind_norm,
    "day_of_year": day_of_year, "is_weekend": is_weekend,
    "season_springer": season_springer, "season_summer": season_summer,
    "season_winter": season_winter,
    "weathersit_Heavy Rain": ws_heavy, "weathersit_Light Snow": ws_snow,
    "weathersit_Mist": ws_mist,
    "workingday_Working Day": wd_flag, "holiday_Yes": hol_flag,
}])[FEATURE_COLS]

# Execution Inference Pipeline Step
if model_loaded:
    NUM_FEATS = ["temp", "atemp", "hum", "windspeed"]
    input_df[NUM_FEATS] = scaler.transform(input_df[NUM_FEATS])
    prediction = max(0, int(round(model.predict(input_df)[0])))
else:
    prediction = None

# ── Dynamic Layout View Panels ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Demand Engine Inference", "📊 Model Architecture Benchmark", "ℹ️ Engineering Documentation"])

# ───────────────────────────────────── TAB 1 ──────────────────────────────────
with tab1:
    if not model_loaded:
        st.error(
            "⚠️ **Execution Halting: Core Inference Drivers Missing.** "
            "Verify dependencies (`best_model.pkl`, `scaler.pkl`, `model_results.json`) are generated "
            "and positioned relative to execution directory path root."
        )
    else:
        # Volumetric Quantile Classification Rules
        if prediction < 50:
            lvl, badge, bar_color = "Low Capacity Required", "🟢", "#2ecc71"
        elif prediction < 150:
            lvl, badge, bar_color = "Moderate Volumetric Load", "🟡", "#f1c40f"
        elif prediction < 350:
            lvl, badge, bar_color = "High Operational Demand", "🟠", "#e67e22"
        else:
            lvl, badge, bar_color = "Critical Capacity Threshold", "🔴", "#e74c3c"

        # Fleet Scheduling Shift Encodings
        if   7 <= hour <= 9:   period = "Morning Rush Commute Peak"
        elif 17 <= hour <= 19: period = "Evening Rush Commute Peak"
        elif 10 <= hour <= 16: period = "Core Midday Operational Window"
        elif 20 <= hour <= 23: period = "Night Distribution Phase"
        else:                  period = "Low-Activity Off-Peak Cycle"

        # Key Performance Metrics Grid
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Vehicle Volume / Hr", f"{prediction:,}")
        c2.metric("System Load Designation", f"{badge} {lvl}")
        c3.metric("Fleet Management Phase", period)
        c4.metric("Temporal Day Group", selected_date.strftime("%A"))

        st.divider()

        # Dynamic Scale Target Gauge Render
        MAX_SCALE = 800
        pct = min(prediction / MAX_SCALE, 1.0)
        st.markdown("**Dynamic Operational Deployment Gauge** (Scales up to 800 units max load capacity)")
        st.markdown(
            f'<div style="background:#e0e0e0;border-radius:10px;height:32px;overflow:hidden">'
            f'<div style="width:{pct*100:.1f}%;background:{bar_color};height:100%;'
            f'display:flex;align-items:center;padding-left:12px;'
            f'color:white;font-weight:bold;font-size:14px">'
            f'{prediction} Fleet Allocations Active</div></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Input Parameter Reference Grid
        st.subheader("Configured Runtime Evaluation Vector")
        items = [
            ("Date Tracked", selected_date.strftime("%b %d, %Y")),
            ("Hour Step",    f"{hour:02d}:00"),
            ("Season Category", season),
            ("Atmospheric Vector", weather),
            ("Thermal Metric", f"{temp_c:.1f} °C (Feels: {atemp_c:.1f} °C)"),
            ("Relative Saturation", f"{humidity} %"),
            ("Anemometer Output", f"{wind_kmh:.1f} km/h"),
            ("Business Working Day", "True" if workingday else "False"),
            ("Statutory Holiday", "True" if holiday else "False"),
            ("Weekend Sequence", "True" if is_weekend else "False"),
        ]
        cols = st.columns(5)
        for i, (k, v) in enumerate(items):
            cols[i % 5].markdown(f"**{k}** \n`{v}`")

# ───────────────────────────────────── TAB 2 ──────────────────────────────────
with tab2:
    st.subheader("Model Evaluation Strategy & Statistical Evaluation Results")

    results_data = load_evaluation_metrics()
    df_res = pd.DataFrame(results_data)

    best_rmse = df_res["RMSE"].min()
    best_r2   = df_res["R²"].max()

    def _highlight_optimal(row):
        if row["RMSE"] == best_rmse:
            return ["background-color:#d4edda;font-weight:bold;color:#155724"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_res.style
              .apply(_highlight_optimal, axis=1)
              .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Render Matplotlib Comparison Space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    c_rmse = ["#1565C0" if r == best_rmse else "#90CAF9" for r in df_res["RMSE"]]
    ax1.barh(df_res["Model"], df_res["RMSE"], color=c_rmse, edgecolor="white")
    ax1.set_xlabel("RMSE (Lower Value Equals High Precision)")
    ax1.set_title("Root Mean Squared Error (RMSE)")
    ax1.invert_yaxis()
    for bar, val in zip(ax1.patches, df_res["RMSE"]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=8)

    c_r2 = ["#2E7D32" if r == best_r2 else "#A5D6A7" for r in df_res["R²"]]
    ax2.barh(df_res["Model"], df_res["R²"], color=c_r2, edgecolor="white")
    ax2.set_xlabel("R² Variance Score (Scale Limit 1.00)")
    ax2.set_title("Coefficient of Determination (R²)")
    ax2.invert_yaxis()
    ax2.set_xlim(df_res["R²"].min() - 0.02, 1.01)
    for bar, val in zip(ax2.patches, df_res["R²"]):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best_row = df_res[df_res["RMSE"] == best_rmse].iloc[0]
    st.success(
        f"**Optimal Model Identified:** {best_row['Model']} Model — "
        f"Validation Benchmarks: MAE = {best_row['MAE']:.2f}, RMSE = {best_row['RMSE']:.2f}, R² = {best_row['R²']:.4f}"
    )

# ───────────────────────────────────── TAB 3 ──────────────────────────────────
with tab3:
    st.subheader("Project Blueprint & Methodology Summary")
    st.markdown("""
**Data Architecture:** Capital Bikeshare Network — Regional Transport Logs
- **Dataset Dimension:** 17,379 unique continuous hour matrices.
- **Target Optimization Metric:** `cnt` (Sum total distribution vector = Casual users + Registered users).

---

### Machine Learning Framework Implementations & Pipeline Architecture

| Engine Iteration | Mathematical Base Type | Structural Operational Strategy |
|---|---|---|
| 1 | Decision Tree Regressor | Non-parametric decision tree framework splits. |
| 2 | Random Forest Ensemble | Bootstrap aggregation forest with decoupled estimators. |
| 3 | Gradient Boosting Machine | Sequential error-residual gradient corrections. |
| 4 | Ridge Regression | Regularized linear structure with L2 parameter penaltization. |
| 5 | Extra Trees Optimization | Random choice cut-point split evaluations. |
| 6 | XGBoost Engine | High-performance extreme gradient boosted tree matrix computations. |

*Note: Hyperparameter Tuning paths utilize continuous cross-validation (`GridSearchCV` and `RandomizedSearchCV` stratified grids).*

---

### Key Data Insights Discovered
- **Diurnal Signature Peaks:** Peak target concentrations occur reliably during workdays at commuter inflection hours (08:00 and 17:00–18:00).
- **Macro Volumetric Expansion:** Temporal progression highlights high structural expansion trends year-over-year.
- **Thermal Correlation:** Core positive demand values directly map to positive scaling index variations in temperature vectors.
- **Weather Dampening:** Strong precipitation vectors act as strong negative drivers, immediately suppressing field volumes.

---

### File Tree Architecture Prerequisites
```
├── best_model.pkl       # Serialized Optimized Predictor Weights File
├── scaler.pkl           # MinMaxScaler Mathematical Fit Array
├── model_results.json   # Multi-Model Structural Cross-Validation Metric Array
└── app.py               # This Core Analytics Application File
```
""")
