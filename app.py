import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import joblib
from datetime import datetime
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# Data generation function for demonstration
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        # Battery parameters
        'pack_voltage': np.random.uniform(350, 400, n_samples),
        'current': np.random.uniform(-200, 200, n_samples),
        'cell_min_voltage': np.random.uniform(3.0, 3.6, n_samples),
        'cell_max_voltage': np.random.uniform(3.7, 4.2, n_samples),
        'cell_temp': np.random.uniform(20, 45, n_samples),
        'pack_power': np.random.uniform(-75000, 75000, n_samples),
        'latitude': np.random.uniform(40, 42, n_samples),
        'longitude': np.random.uniform(-74, -72, n_samples),
        
        # Charging parameters
        'charge_voltage': np.random.uniform(350, 400, n_samples),
        'charge_current': np.random.uniform(0, 250, n_samples),
        'charge_power': np.random.uniform(0, 100000, n_samples),
        'charge_time': np.random.uniform(0, 60, n_samples),
        'charge_location_type': np.random.choice(['home', 'public', 'supercharger'], n_samples),
        
        # Additional parameters
        'ambient_temp': np.random.uniform(0, 35, n_samples),
        'battery_age_months': np.random.uniform(0, 60, n_samples),
        'state_of_charge': np.random.uniform(0, 100, n_samples),
        'charging_efficiency': np.random.uniform(85, 95, n_samples)
    }
    
    # Target variable: Remaining range prediction
    data['remaining_range'] = (
        data['state_of_charge'] * 0.5 +
        data['charging_efficiency'] * 0.3 +
        np.random.normal(0, 5, n_samples)
    )
    
    return pd.DataFrame(data)

# Model training function
def train_model(df):
    # Prepare features and target
    feature_columns = [
        'pack_voltage', 'current', 'cell_min_voltage', 'cell_max_voltage',
        'cell_temp', 'pack_power', 'charge_voltage', 'charge_current',
        'charge_power', 'charge_time', 'ambient_temp', 'battery_age_months',
        'state_of_charge', 'charging_efficiency'
    ]
    
    X = df[feature_columns]
    y = df['remaining_range']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, feature_columns

# Streamlit app
def main():
    st.title("EV Battery Analysis and Prediction System")
    
    # Sidebar
    st.sidebar.header("Model Training")
    if st.sidebar.button("Train New Model"):
        with st.spinner("Generating sample data and training model..."):
            df = generate_sample_data()
            model, scaler, mse, r2, feature_columns = train_model(df)
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_columns'] = feature_columns
            st.sidebar.success(f"Model trained successfully!\nMSE: {mse:.2f}\nR² Score: {r2:.2f}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Battery Monitoring", "Charge Station Analysis", "Predictions"])
    
    with tab1:
        st.header("Battery Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pack_voltage = st.number_input("Pack Voltage (V)", 350.0, 400.0, 375.0)
            current = st.number_input("Current (A)", -200.0, 200.0, 0.0)
            cell_min_voltage = st.number_input("Cell Min Voltage (V)", 3.0, 3.6, 3.3)
        
        with col2:
            cell_max_voltage = st.number_input("Cell Max Voltage (V)", 3.7, 4.2, 4.0)
            cell_temp = st.number_input("Cell Temperature (°C)", 20.0, 45.0, 30.0)
            pack_power = st.number_input("Pack Power (W)", -75000.0, 75000.0, 0.0)
        
        with col3:
            ambient_temp = st.number_input("Ambient Temperature (°C)", 0.0, 35.0, 25.0)
            battery_age = st.number_input("Battery Age (months)", 0, 60, 12)
            soc = st.number_input("State of Charge (%)", 0.0, 100.0, 80.0)
    
    with tab2:
        st.header("Charging Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            charge_voltage = st.number_input("Charging Voltage (V)", 350.0, 400.0, 375.0)
            charge_current = st.number_input("Charging Current (A)", 0.0, 250.0, 100.0)
            charge_power = st.number_input("Charging Power (W)", 0.0, 100000.0, 50000.0)
        
        with col2:
            charge_time = st.number_input("Charging Time (min)", 0.0, 60.0, 30.0)
            charge_location = st.selectbox("Charging Location", ['home', 'public', 'supercharger'])
            charging_efficiency = st.number_input("Charging Efficiency (%)", 85.0, 95.0, 90.0)
    
    with tab3:
        st.header("Range Prediction")
        if st.button("Predict Range"):
            if 'model' not in st.session_state:
                st.warning("Please train the model first using the sidebar button!")
            else:
                # Prepare input data for prediction
                input_data = pd.DataFrame([[
                    pack_voltage, current, cell_min_voltage, cell_max_voltage,
                    cell_temp, pack_power, charge_voltage, charge_current,
                    charge_power, charge_time, ambient_temp, battery_age,
                    soc, charging_efficiency
                ]], columns=st.session_state['feature_columns'])
                
                # Scale input data and predict
                input_scaled = st.session_state['scaler'].transform(input_data)
                prediction = st.session_state['model'].predict(input_scaled)[0]
                
                st.success(f"Predicted Remaining Range: {prediction:.2f} km")
                
                # Feature importance plot
                importance = pd.DataFrame({
                    'feature': st.session_state['feature_columns'],
                    'importance': st.session_state['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(importance, x='importance', y='feature', 
                            title='Feature Importance',
                            orientation='h')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()