
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from src.pipeline import (
    FEATURES,
    load_dataset,
    load_or_train,
    train_models,
    build_prediction_row,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="GridPulse | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# ALERT CONFIGURATION
# =====================================================

# Operational early-warning threshold.
# This does NOT change the trained model's accuracy.
ALERT_THRESHOLD = 0.40

# =====================================================
# PREMIUM LIGHT THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background: #f3f6fc;
    color: #17243c;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: #14213d !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111c35, #1d3155);
}

[data-testid="stSidebar"] * {
    color: #f1f5ff !important;
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #ffffff12;
    border: 1px solid #ffffff25;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e8f3;
    border-radius: 17px;
    padding: 19px;
    box-shadow: 0 5px 20px #18345b0b;
}

[data-testid="stMetricLabel"] {
    color: #60718e !important;
}

[data-testid="stMetricValue"] {
    color: #14213d !important;
    font-weight: 800;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: white !important;
    color: #17243c !important;
    border-color: #ccd8e9 !important;
    border-radius: 10px !important;
}

[data-testid="stWidgetLabel"] p {
    color: #263b5b !important;
    font-weight: 700;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(110deg, #2563eb, #7656e8);
    color: white !important;
    border: none;
    border-radius: 11px;
    min-height: 43px;
    font-weight: 800;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(110deg, #174ec5, #6138d3);
    color: white !important;
    border: none;
    box-shadow: 0 5px 18px #526ce455;
}

.hero {
    background: linear-gradient(115deg, #173c79, #4b45a7, #087d91);
    padding: 32px;
    border-radius: 23px;
    color: white;
    margin-bottom: 27px;
    box-shadow: 0 12px 30px #243e7025;
}

.hero * {
    color: white !important;
}

.hero-label {
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 800;
    opacity: .85;
}

.hero-title {
    font-size: 39px;
    font-family: 'Space Grotesk';
    font-weight: 800;
    margin-top: 8px;
}

.hero-desc {
    font-size: 14px;
    opacity: .9;
    margin-top: 8px;
}

.hero-tag {
    display: inline-block;
    background: #ffffff20;
    border: 1px solid #ffffff50;
    border-radius: 20px;
    padding: 6px 12px;
    font-size: 11px;
    margin: 15px 5px 0 0;
}

[data-testid="stDataFrame"] {
    background: white;
    border: 1px solid #dce5f2;
    border-radius: 12px;
}

hr {
    border-color: #dce5f2;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA AND MODEL
# =====================================================

@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Loading GridPulse ML model...")
def get_model():
    return load_or_train()


try:
    df = get_dataset()
    artifact, report = get_model()

except Exception as e:
    st.error(f"Project loading failed: {e}")
    st.info("Check data/electricity_demand.csv and src/pipeline.py.")
    st.stop()


# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">
        TEAM 11 · MACHINE LEARNING PROJECT
    </div>
    <div class="hero-title">⚡ GridPulse</div>
    <div style="font-size:20px;font-weight:700;">
        Electricity Demand Spike Alert System
    </div>
    <div class="hero-desc">
        Smart demand monitoring, scenario analysis
        and machine learning based spike risk prediction.
    </div>
    <span class="hero-tag">ML Intelligence</span>
    <span class="hero-tag">Demand Analytics</span>
    <span class="hero-tag">Live Scenario</span>
    <span class="hero-tag">Team 11</span>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# ⚡ GRIDPULSE")
    st.caption("ENERGY INTELLIGENCE")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Live Demand Monitor",
            "Model Intelligence",
            "Dataset Explorer",
        ],
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")
    st.success("Dataset connected")
    st.success("ML model loaded")

    st.metric("Dataset Records", f"{len(df):,}")
    st.metric("Features", len(FEATURES))

    st.divider()
    st.markdown("### MODEL MANAGEMENT")

    if st.button("🔄 Retrain Models", width="stretch"):
        try:
            with st.spinner("Training ML models..."):
                train_models(df)

            get_model.clear()
            st.success("Training completed!")
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")

    st.divider()
    st.caption("GridPulse · Team 11")
    st.caption("Academic demonstration prototype")


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.markdown("## Dashboard Overview")
    st.caption("Monitor your electricity demand dataset.")

    metrics = report["selected_metrics"]

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Records", f"{len(df):,}")

    k2.metric(
        "Average Demand",
        f"{df['current_demand_mw'].mean():.2f} MW",
    )

    k3.metric(
        "Spike Records",
        f"{int(df['spike_next_hour'].sum()):,}",
    )

    k4.metric("Model F1 Score", f"{metrics['f1']:.3f}")

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("### 📈 Electricity Demand Trend")

        history = df.tail(168).copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])

        st.area_chart(
            history.set_index("timestamp")[["current_demand_mw"]],
            color="#3185f5",
            height=330,
        )

        st.caption("Last 168 dataset records")

    with right:
        st.markdown("### 📊 Normal vs Spike")

        counts = (
            df["spike_next_hour"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        status_df = pd.DataFrame(
            {"Records": counts.values},
            index=["Normal", "Spike"],
        )

        st.bar_chart(status_df, color="#8261e8", height=330)

    st.markdown("### Recent Demand Records")

    recent = df.tail(15).copy()

    recent["Status"] = recent["spike_next_hour"].map(
        {0: "NORMAL", 1: "SPIKE"}
    )

    st.dataframe(
        recent[
            [
                "timestamp",
                "current_demand_mw",
                "temperature_c",
                "humidity_pct",
                "Status",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


# =====================================================
# LIVE DEMAND MONITOR
# =====================================================

elif page == "Live Demand Monitor":

    st.markdown("## ⚡ Live Demand Monitor")

    st.caption(
        "Change input values to update the ML prediction and scenario table."
    )

    latest = df.iloc[-1]

    st.markdown("### Input Control Centre")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour = st.number_input(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=int(latest["hour"]),
            step=1,
        )

        day = st.number_input(
            "Day of Week (0 = Monday)",
            min_value=0,
            max_value=6,
            value=int(latest["day_of_week"]),
            step=1,
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=55.0,
            value=float(latest["temperature_c"]),
            step=0.5,
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(latest["humidity_pct"]),
            step=1.0,
        )

    with col3:
        demand = st.number_input(
            "Current Demand (MW)",
            min_value=50.0,
            max_value=250.0,
            value=float(latest["current_demand_mw"]),
            step=1.0,
        )

    input_values = {
        "hour": hour,
        "day_of_week": day,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "current_demand_mw": demand,
    }

    try:

        input_row = build_prediction_row(input_values, df)

        model = artifact["model"]

        # Original ML threshold from validation
        model_threshold = float(artifact["threshold"])

        # Actual model probability
        probability = float(
            model.predict_proba(input_row[FEATURES])[0][1]
        )

        # Operational alert decision
        prediction = int(probability >= ALERT_THRESHOLD)

        st.divider()
        st.markdown("### 🎯 Prediction Results")

        p1, p2, p3 = st.columns(3)

        p1.metric("Current Demand", f"{demand:.2f} MW")
        p2.metric("Spike Probability", f"{probability:.1%}")

        p3.metric(
            "Prediction",
            "HIGH RISK" if prediction else "NORMAL",
        )

        if prediction:
            st.error(
                f"🚨 HIGH DEMAND RISK — Probability: {probability:.1%}"
            )
        else:
            st.success(
                f"✅ NORMAL DEMAND — Probability: {probability:.1%}"
            )

        # -------------------------------------------------
        # DYNAMIC TABLE
        # -------------------------------------------------

        st.markdown("### 📋 Dynamic Demand Table")

        historical = df.tail(12).copy()

        historical_table = pd.DataFrame({
            "Timestamp": historical["timestamp"].astype(str),
            "Demand (MW)": historical["current_demand_mw"],
            "Temperature (°C)": historical["temperature_c"],
            "Humidity (%)": historical["humidity_pct"],
            "Status": historical["spike_next_hour"].map(
                {0: "NORMAL", 1: "SPIKE"}
            ),
            "Record": "Historical",
        })

        next_time = (
            pd.to_datetime(df["timestamp"].max())
            + pd.Timedelta(hours=1)
        )

        live_table = pd.DataFrame([{
            "Timestamp": next_time.strftime("%Y-%m-%d %H:%M"),
            "Demand (MW)": demand,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Status": "HIGH RISK" if prediction else "NORMAL",
            "Record": "LIVE INPUT",
        }])

        final_table = pd.concat(
            [historical_table, live_table],
            ignore_index=True,
        )

        st.dataframe(
            final_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Demand (MW)": st.column_config.NumberColumn(
                    format="%.2f MW"
                ),
                "Temperature (°C)": st.column_config.NumberColumn(
                    format="%.1f °C"
                ),
                "Humidity (%)": st.column_config.NumberColumn(
                    format="%.1f %%"
                ),
            },
        )

        st.caption(
            "LIVE INPUT updates when you change the input controls. "
            "Historical records remain unchanged."
        )

        # -------------------------------------------------
        # COMPARISON CHART
        # -------------------------------------------------

        st.markdown("### 📈 Demand Comparison")

        comparison = pd.DataFrame(
            {
                "Demand (MW)": [
                    float(df.iloc[-1]["current_demand_mw"]),
                    demand,
                    float(df.iloc[-1]["demand_lag_1h_mw"]),
                ]
            },
            index=[
                "Latest Dataset Demand",
                "Your Input",
                "Previous Hour Demand",
            ],
        )

        st.bar_chart(
            comparison,
            color="#3988ed",
            height=300,
        )

        # -------------------------------------------------
        # DECISION DETAILS
        # -------------------------------------------------

        st.markdown("### Model Decision Details")

        st.progress(probability)

        st.write(f"Spike probability: **{probability:.1%}**")
        st.write(
            f"Operational alert threshold: **{ALERT_THRESHOLD:.0%}**"
        )
        st.write(
            f"Original ML threshold: **{model_threshold:.3f}**"
        )

        if prediction:
            st.warning(
                "Input crossed the operational early-warning threshold. "
                "Review the demand conditions."
            )
        else:
            st.info(
                "Input is below the operational early-warning threshold."
            )

        st.caption(
            "HIGH RISK is an operational early-warning label based on a "
            "40% probability threshold. It is not confirmation of an "
            "actual electricity demand spike. This dashboard uses "
            "user-entered scenario data, not live grid sensor data."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


# =====================================================
# MODEL INTELLIGENCE
# =====================================================

elif page == "Model Intelligence":

    st.markdown("## 🧠 Model Intelligence")
    st.caption("Machine learning model evaluation and performance.")

    comparison = pd.DataFrame(report["test_results"]).T
    comparison.index.name = "Algorithm"

    st.dataframe(comparison, width="stretch")

    selected = report["selected_metrics"]

    st.markdown("### Selected Model")

    a, b, c, d = st.columns(4)

    a.metric("Model", artifact["model_name"])
    b.metric("Accuracy", f"{selected['accuracy']:.3f}")
    c.metric("Precision", f"{selected['precision']:.3f}")
    d.metric("Recall", f"{selected['recall']:.3f}")

    st.markdown("### Confusion Matrix")

    cm = np.asarray(selected["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Spike"],
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )

    ax.set_title("Model Evaluation")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.tick_params(colors="#17243c")
    ax.xaxis.label.set_color("#17243c")
    ax.yaxis.label.set_color("#17243c")

    for spine in ax.spines.values():
        spine.set_color("#ccd8e9")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Dataset Split")

    x, y, z = st.columns(3)

    x.metric("Training Records", f"{report['train_rows']:,}")
    y.metric("Validation Records", f"{report['validation_rows']:,}")
    z.metric("Testing Records", f"{report['test_rows']:,}")

    st.caption(
        "Chronological split: training, validation and testing."
    )

    st.markdown("### Alert Configuration")

    st.metric(
        "Operational Early-Warning Threshold",
        f"{ALERT_THRESHOLD:.0%}",
    )

    st.caption(
        "This operational threshold is separate from model evaluation."
    )


# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.markdown("## 🗂️ Dataset Explorer")

    a, b, c, d = st.columns(4)

    a.metric("Total Records", f"{len(df):,}")
    b.metric("Missing Values", f"{df.isna().sum().sum():,}")

    c.metric(
        "Duplicate Timestamps",
        f"{df['timestamp'].duplicated().sum():,}",
    )

    d.metric(
        "Spike Records",
        f"{int(df['spike_next_hour'].sum()):,}",
    )

    st.markdown("### Filter Records")

    category = st.selectbox(
        "Demand Category",
        ["All", "Normal", "Spike"],
    )

    filtered = df.copy()

    if category == "Normal":
        filtered = filtered[filtered["spike_next_hour"] == 0]

    elif category == "Spike":
        filtered = filtered[filtered["spike_next_hour"] == 1]

    st.caption(f"Displaying {len(filtered):,} records")

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="gridpulse_filtered_data.csv",
        mime="text/csv",
        width="stretch",
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#64748b;font-size:12px;">
    ⚡ GRIDPULSE · TEAM 11<br>
    Electricity Demand Spike Alert Using Machine Learning<br>
    Academic Demonstration Project
</div>
""", unsafe_allow_html=True)
