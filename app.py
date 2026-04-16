import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Uber Fare Predictor", page_icon="🚕", layout="centered")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🚕 Uber Fare Predictor")
st.write("Estimate your cab fare with smart pricing")

# Inputs
distance = st.number_input("Distance (km)", min_value=0.0, step=0.5)
passengers = st.number_input("Passengers", min_value=1, max_value=6)
hour = st.slider("Hour of Day", 0, 23, 12)

# Prediction
if st.button("Estimate Fare"):

    # Prepare input
    input_data = pd.DataFrame([[distance, passengers, hour]],
                              columns=['distance', 'passenger_count', 'hour'])

    # 🔹 Step 1: ML Prediction
    base_fare = model.predict(input_data)[0]

    # 🔹 Step 2: Scale model output (important)
    fare = base_fare * 20

    # 🔹 Step 3: Distance correction (India pricing)
    fare += distance * 8

    # 🔹 Step 4: Passenger effect
    fare += passengers * 5

    # 🔹 Step 5: Night charges (10 PM – 6 AM)
    if hour >= 22 or hour <= 6:
        fare *= 1.15

    # 🔹 Step 6: Peak hours (8–10 AM, 5–8 PM)
    if (8 <= hour <= 10) or (17 <= hour <= 20):
        fare *= 1.1

    # 🔹 Step 7: Minimum fare
    if fare < 60:
        fare = 60

    # 🔹 Output
    st.success(f"Estimated Fare: ₹ {fare:.2f}")

    # Optional: show model output (for understanding)
    st.caption(f"Base Model Output: {base_fare:.2f}")