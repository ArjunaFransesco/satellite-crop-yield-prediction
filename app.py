import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Satellite NDVI & Agro-Climatic Crop Yield Forecaster",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Satellite NDVI & Agro-Climatic Crop Yield Forecaster")
st.markdown("**Domain**: `AgriTech / Geospatial ML` | **Tech Stack**: `LightGBM, Random Forest, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    ndvi_vegetation_index = st.slider("Ndvi Vegetation Index", float(0.1), float(0.95), float(0.68))
    mean_surface_temp_c = st.slider("Mean Surface Temp C", float(12.0), float(42.0), float(26.5))
    cumulative_rainfall_mm = st.slider("Cumulative Rainfall Mm", float(100.0), float(1600.0), float(680.0))
    soil_moisture_pct = st.slider("Soil Moisture Pct", float(5.0), float(65.0), float(34.0))
    elevation_meters = st.slider("Elevation Meters", float(10.0), float(1200.0), float(280.0))
    solar_radiation_index = st.slider("Solar Radiation Index", float(8.0), float(28.0), float(18.5))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"ndvi_vegetation_index": ndvi_vegetation_index, "mean_surface_temp_c": mean_surface_temp_c, "cumulative_rainfall_mm": cumulative_rainfall_mm, "soil_moisture_pct": soil_moisture_pct, "elevation_meters": elevation_meters, "solar_radiation_index": solar_radiation_index}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `crop_yield_ton_per_ha`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
