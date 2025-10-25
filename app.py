import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
# model = joblib.load("aqi_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Air Quality Index (AQI) Prediction")
st.markdown("Enter pollutant levels below to predict the AQI.")

# -----------------------------
# Pollutant Defaults (min, max, default)
# -----------------------------
pollutant_defaults = {
    "PM2.5": (0, 500, 35),
    "PM10": (0, 600, 50),
    "NO": (0, 200, 15),
    "NO2": (0, 200, 20),
    "NOx": (0, 300, 30),
    "NH3": (0, 200, 12),
    "CO": (0, 20, 0.8),
    "SO2": (0, 200, 5),
    "O3": (0, 300, 10),
    "Benzene": (0, 50, 1.2),
    "Toluene": (0, 50, 0.5),
    "Xylene": (0, 50, 0.3)
}

# -----------------------------
# Input Fields in 2 Columns (all float)
# -----------------------------
inputs = {}
pollutants = list(pollutant_defaults.keys())

for i in range(0, len(pollutants), 2):
    col1, col2 = st.columns(2)
    
    # First column
    p1 = pollutants[i]
    min1, max1, default1 = map(float, pollutant_defaults[p1])
    inputs[p1] = col1.number_input(
        p1, min_value=min1, max_value=max1, value=default1, format="%.2f"
    )
    
    # Second column
    if i + 1 < len(pollutants):
        p2 = pollutants[i + 1]
        min2, max2, default2 = map(float, pollutant_defaults[p2])
        inputs[p2] = col2.number_input(
            p2, min_value=min2, max_value=max2, value=default2, format="%.2f"
        )

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict AQI"):
    # Prepare input for model
    feature_array = np.array([[v for v in inputs.values()]])
    
    # Predict AQI
    prediction = model.predict(feature_array)[0]
    
    # Color-coded AQI status
    if prediction <= 50:
        status = "Good 🌿"
        color = "#2ECC71"
    elif prediction <= 100:
        status = "Moderate ⚠️"
        color = "#F1C40F"
    elif prediction <= 200:
        status = "Poor 😷"
        color = "#E67E22"
    else:
        status = "Very Poor ☠️"
        color = "#E74C3C"
    
    # Display prediction beautifully
    st.markdown(f"""
        <div style="background-color:{color};padding:25px;border-radius:15px;text-align:center">
            <h2 style="color:white;margin:0;">Predicted AQI: {prediction:.2f}</h2>
            <h3 style="color:white;margin:0;">Air Quality: {status}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Display pollutant bar chart
    st.subheader("Pollutant Levels")
    df_pollutants = pd.DataFrame.from_dict(inputs, orient='index', columns=['Value'])
    st.bar_chart(df_pollutants)
