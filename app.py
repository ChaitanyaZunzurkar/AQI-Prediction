import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load("./ML_Model/aqi_prediction.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🌿 AQI Prediction App",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Air Quality Index (AQI) Prediction")
st.markdown("### Enter pollutant concentrations below to predict the AQI.")

# -----------------------------
# Pollutant Ranges (min, max, default)
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

# -----------------------------
# Collect User Inputs
# -----------------------------
inputs = {}
pollutants = list(pollutant_defaults.keys())

st.markdown("#### Enter pollutant levels:")

for i in range(0, len(pollutants), 2):
    col1, col2 = st.columns(2)
    
    # Column 1
    p1 = pollutants[i]
    min1, max1, default1 = pollutant_defaults[p1]
    inputs[p1] = col1.number_input(
        f"{p1} (µg/m³)", 
        min_value=min1, 
        max_value=max1, 
        value=default1, 
        step=0.1
    )

    # Column 2
    if i + 1 < len(pollutants):
        p2 = pollutants[i + 1]
        min2, max2, default2 = pollutant_defaults[p2]
        inputs[p2] = col2.number_input(
            f"{p2} (µg/m³)", 
            min_value=min2, 
            max_value=max2, 
            value=default2, 
            step=0.1
        )

# -----------------------------
# Predict Button
# -----------------------------
st.markdown("---")
if st.button("🚀 Predict AQI"):
    try:
        # Prepare input for model
        input_df = pd.DataFrame([inputs])  # Keeps feature names
        prediction = model.predict(input_df)[0]

        prediction = round(float(prediction), 2)

        # Determine AQI Category
        if prediction <= 50:
            status, color = "Good 🌿", "#2ECC71"
        elif prediction <= 100:
            status, color = "Moderate ⚠️", "#F1C40F"
        elif prediction <= 200:
            status, color = "Poor 😷", "#E67E22"
        else:
            status, color = "Very Poor ☠️", "#E74C3C"

        # Display result
        st.markdown(f"""
            <div style="background-color:{color};
                        padding:25px;
                        border-radius:15px;
                        text-align:center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                <h2 style="color:white;">Predicted AQI: {prediction}</h2>
                <h3 style="color:white;">Air Quality: {status}</h3>
            </div>
        """, unsafe_allow_html=True)

        # Pollutant Chart
        st.markdown("### 📊 Pollutant Levels Overview")
        df_pollutants = pd.DataFrame(inputs.items(), columns=["Pollutant", "Value"]).set_index("Pollutant")
        st.bar_chart(df_pollutants)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

