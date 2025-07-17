import streamlit as st
import pandas as pd
import joblib
import datetime

# Load trained model
model = joblib.load('corrosion_model.pkl')

st.title("🛢️ Fluid Guard - AI Corrosion Predictor")
st.write("Estimate days until corrosion starts based on pipeline & fluid data.")

# User inputs
material = st.selectbox("Pipeline Material", ["Steel", "Iron", "PVC", "Aluminium", "Copper"])
fluid_type = st.selectbox("Fluid Type", ["Water", "Crude Oil", "Petrol", "Diesel", "Kerosene", "Other"])
soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Rocky", "Other"])
install_date = st.date_input("Pipeline Installation Date", value=datetime.date(2018, 1, 1))
age_years = st.number_input("Age (years)", 0, 50, 5)
diameter_mm = st.number_input("Diameter (mm)", 50, 2000, 500)
length_km = st.number_input("Length (km)", 1, 1000, 10)
fluid_temp_C = st.number_input("Fluid Temperature (°C)", -10, 150, 40)
pressure_bar = st.number_input("Fluid Pressure (bar)", 1, 100, 20)
density_kgm3 = st.number_input("Fluid Density (kg/m³)", 500, 1500, 900)
humidity_pct = st.slider("Humidity (%)", 0, 100, 60)
flow_rate_lph = st.number_input("Flow Rate (L/h)", 100, 100000, 5000)
ambient_temp_C = st.number_input("Ambient Temperature (°C)", -20, 50, 30)
external_pressure_bar = st.number_input("External Pressure (bar)", 0, 10, 2)
maint_interval = st.slider("Maintenance Interval (days)", 30, 365, 180)
flow_freq = st.slider("Flow Frequency (hours/day)", 0, 24, 12)

# Prepare input for model
input_dict = {
    'age_years': age_years,
    'diameter_mm': diameter_mm,
    'length_km': length_km,
    'fluid_temp_C': fluid_temp_C,
    'pressure_bar': pressure_bar,
    'density_kgm3': density_kgm3,
    'humidity_pct': humidity_pct,
    'flow_rate_lph': flow_rate_lph,
    'ambient_temp_C': ambient_temp_C,
    'external_pressure_bar': external_pressure_bar,
    f"material_{material}": 1,
    f"fluid_type_{fluid_type}": 1,
    f"soil_type_{soil_type}": 1
}

# Fill missing columns with zeros
expected_cols = model.feature_names_in_
input_df = pd.DataFrame([input_dict])
for col in expected_cols:
    if col not in input_df:
        input_df[col] = 0
input_df = input_df[expected_cols]

if st.button("Predict Days to Corrosion"):
    # --- ML Prediction ---
    pred_days = int(model.predict(input_df)[0])
    st.subheader(f"🧪 Estimated days to corrosion: {pred_days} days")

    # --- Additional logic: corrosion & replacement dates ---
    today = datetime.date.today()
    corrosion_date = install_date + datetime.timedelta(days=pred_days)
    replacement_date = install_date + datetime.timedelta(days=int(pred_days * 2.5))

    maint_ok = (maint_interval < (pred_days // 4))
    suggested_interval = max(30, pred_days // 5)

    st.header("📅 Predictions")
    st.write(f"🔶 *Estimated Corrosion Starts:* {corrosion_date.strftime('%B %Y')}")
    st.write(f"🔴 *Recommended Pipeline Replacement:* {replacement_date.strftime('%B %Y')}")

    if maint_ok:
        st.success("✅ Your maintenance schedule is adequate.")
    else:
        st.warning("⚠ Maintenance interval too long. Consider reducing to avoid early rust.")
        st.info(f"📉 Suggested Maintenance Interval: Every {suggested_interval} days")

    # --- Flow frequency advice ---
    st.header("💡 Suggestions")
    if flow_freq > 16:
        st.warning("📛 High flow frequency detected. Consider reducing to minimize wear and corrosion risk.")
    elif flow_freq < 6:
        st.info("🔁 Low flow frequency. This is generally safer, but verify fluid stagnation is not a risk.")
    else:
        st.success("✅ Flow frequency is within a safe operating range.")

    st.markdown("---")
    st.caption("Fluid Guard - Built for predictive pipeline care in Nigeria and beyond. 🚀")
