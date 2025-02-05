import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load the models and scaler
@st.cache_resource
def load_models():
    with open('cost_model.pkl', 'rb') as f:
        cost_model = pickle.load(f)
    with open('distance_model.pkl', 'rb') as f:
        distance_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return cost_model, distance_model, scaler

def main():
    st.title("EV Charging Predictor")
    st.write("Predict charging costs and distance driven for electric vehicles")
    
    cost_model, distance_model, scaler = load_models()
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_model = st.selectbox("Vehicle Model", 
            ["BMW i3", "Chevy Bolt", "Hyundai Kona", "Nissan Leaf", "Tesla Model 3"])
        battery_capacity = st.number_input("Battery Capacity (kWh)", 40.0, 150.0, 75.0)
        charging_duration = st.number_input("Charging Duration (hours)", 0.1, 10.0, 2.0)
        charging_rate = st.number_input("Charging Rate (kW)", 1.0, 50.0, 20.0)
        energy_consumed = st.number_input("Energy Consumed (kWh)", 1.0, 100.0, 30.0)
        temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 20.0)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (years)", 0.0, 10.0, 2.0)
        soc_start = st.number_input("State of Charge Start (%)", 0.0, 100.0, 20.0)
        soc_end = st.number_input("State of Charge End (%)", 0.0, 100.0, 80.0)
        location = st.selectbox("Location", 
            ["San Francisco", "Los Angeles", "New York", "Chicago", "Houston"])
        charger_type = st.selectbox("Charger Type", 
            ["Level 1", "Level 2", "DC Fast Charger"])
        user_type = st.selectbox("User Type", 
            ["Commuter", "Casual Driver", "Long-Distance Traveler"])
    
    if st.button("Predict"):
        # Prepare input data
        current_time = datetime.now()
        input_data = {
            'Battery Capacity (kWh)': battery_capacity,
            'Charging Duration (hours)': charging_duration,
            'Charging Rate (kW)': charging_rate,
            'Energy Consumed (kWh)': energy_consumed,
            'Temperature (°C)': temperature,
            'Vehicle Age (years)': vehicle_age,
            'State of Charge (Start %)': soc_start,
            'State of Charge (End %)': soc_end,
            'Hour': current_time.hour,
            'Month': current_time.month,
            'Day': current_time.day,
            'Vehicle Model_encoded': ["BMW i3", "Chevy Bolt", "Hyundai Kona", "Nissan Leaf", "Tesla Model 3"].index(vehicle_model),
            'Charging Station Location_encoded': ["San Francisco", "Los Angeles", "New York", "Chicago", "Houston"].index(location),
            'Time of Day_encoded': 0,  # Simplified for demo
            'Day of Week_encoded': current_time.weekday(),
            'Charger Type_encoded': ["Level 1", "Level 2", "DC Fast Charger"].index(charger_type),
            'User Type_encoded': ["Commuter", "Casual Driver", "Long-Distance Traveler"].index(user_type)
        }
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Make predictions
        cost_pred = cost_model.predict(input_scaled)[0]
        distance_pred = distance_model.predict(input_scaled)[0]
        
        # Display results
        st.success(f"Predicted Charging Cost: ${cost_pred:.2f}")
        st.success(f"Predicted Distance Driven: {distance_pred:.1f} km")
        
        # Provide feedback based on predictions
        st.subheader("Usage Analysis")
        
        if cost_pred > 30:
            st.warning("⚠️ High charging cost predicted. Consider charging during off-peak hours.")
        else:
            st.info("✓ Charging cost is within normal range.")
            
        if distance_pred < 50:
            st.info("Short-distance usage pattern detected - ideal for urban commuting.")
        elif distance_pred > 200:
            st.warning("Long-distance travel pattern - ensure charging stations availability on route.")
        else:
            st.info("Medium-distance usage pattern - good balance of range and charging needs.")
            
        # Efficiency tips based on user type
        st.subheader("Personalized Tips")
        if user_type == "Commuter":
            st.info("💡 Consider scheduling regular charging during off-peak hours for cost savings.")
        elif user_type == "Long-Distance Traveler":
            st.info("💡 Plan your routes around DC Fast Charging stations for optimal charging time.")
        else:
            st.info("💡 Monitor your charging patterns to optimize cost and convenience.")

if __name__ == "__main__":
    main()