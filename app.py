import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🌿 AQI Prediction Dashboard",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load("./ML_Model/aqi_prediction.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
            font-family: 'Poppins', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #2E7D32;
            font-size: 2.8rem;
            font-weight: 700;
            margin-top: -10px;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #1B5E20;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }
        .aql-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            color: white;
            transition: transform 0.3s ease;
        }
        .aql-card:hover {
            transform: scale(1.02);
        }
        .predict-btn button {
            background: linear-gradient(135deg, #43a047, #2e7d32);
            color: white !important;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 20px;
            width: 100%;
        }
        .pollutant-input label {
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/252/252035.png", width=100)
st.sidebar.title("🌿 AQI Predictor")
st.sidebar.markdown("""
### How it works
Enter pollutant concentrations to estimate **Air Quality Index (AQI)**  
and its corresponding **health impact** category.

**AQI Ranges:**
- 🟢 0–50 → Good  
- 🟡 51–100 → Moderate  
- 🟠 101–200 → Poor  
- 🔴 201+ → Very Poor  
""")

# -----------------------------
# Main Header
# -----------------------------
st.markdown('<h1 class="main-title">Air Quality Index Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict the AQI level based on real-time pollutant data</p>', unsafe_allow_html=True)

# -----------------------------
# Pollutant Input Section
# -----------------------------
pollutant_defaults = {
    "PM2.5": (0.0, 500.0, 35.0),
    "PM10": (0.0, 600.0, 50.0),
    "NO": (0.0, 200.0, 15.0),
    "NO2": (0.0, 200.0, 20.0),
    "NOx": (0.0, 300.0, 30.0),
    "NH3": (0.0, 200.0, 12.0),
    "CO": (0.0, 20.0, 0.8),
    "SO2": (0.0, 200.0, 5.0),
    "O3": (0.0, 300.0, 10.0),
    "Benzene": (0.0, 50.0, 1.2),
    "Toluene": (0.0, 50.0, 0.5),
    "Xylene": (0.0, 50.0, 0.3)
}

inputs = {}
pollutants = list(pollutant_defaults.keys())

st.markdown("### 🧪 Input Pollutant Concentrations")

for i in range(0, len(pollutants), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        if i + j < len(pollutants):
            p = pollutants[i + j]
            min_val, max_val, default = pollutant_defaults[p]
            inputs[p] = col.number_input(
                f"{p} (µg/m³)",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=0.1,
                key=p
            )

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict = st.button("🚀 Predict AQI", use_container_width=True)

if predict:
    try:
        input_df = pd.DataFrame([inputs])
        prediction = round(float(model.predict(input_df)[0]), 2)

        # AQI Category
        if prediction <= 50:
            status, color, emoji = "Good", "#2ECC71", "🟢"
        elif prediction <= 100:
            status, color, emoji = "Moderate", "#F1C40F", "🟡"
        elif prediction <= 200:
            status, color, emoji = "Poor", "#E67E22", "🟠"
        else:
            status, color, emoji = "Very Poor", "#E74C3C", "🔴"

        # AQI Card
        st.markdown(f"""
            <div class="aql-card" style="background-color:{color};">
                <h2 style="font-size:2.5rem;">{emoji} Predicted AQI: {prediction}</h2>
                <h3 style="font-weight:600;">Air Quality: {status}</h3>
                <p style="margin-top:10px; font-size:1rem;">This reflects the current air pollution level based on entered pollutant concentrations.</p>
            </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Pollutant Chart
        # -----------------------------
        st.markdown("### 📊 Pollutant Concentration Overview")
        df_pollutants = pd.DataFrame(inputs.items(), columns=["Pollutant", "Value"])
        fig = px.bar(
            df_pollutants,
            x="Pollutant",
            y="Value",
            color="Value",
            color_continuous_scale="Viridis",
            text_auto=True,
            title="Pollutant Concentrations (µg/m³)"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1B5E20", size=14),
            title_x=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
