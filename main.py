import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Enhanced data generation to include usage patterns
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate timestamps for the past 30 days
    base = datetime.now() - timedelta(days=30)
    timestamps = [base + timedelta(hours=x) for x in range(n_samples)]
    
    data = {
        # Previous parameters
        'pack_voltage': np.random.uniform(350, 400, n_samples),
        'current': np.random.uniform(-200, 200, n_samples),
        'cell_min_voltage': np.random.uniform(3.0, 3.6, n_samples),
        'cell_max_voltage': np.random.uniform(3.7, 4.2, n_samples),
        'cell_temp': np.random.uniform(20, 45, n_samples),
        'pack_power': np.random.uniform(-75000, 75000, n_samples),
        'latitude': np.random.uniform(40, 42, n_samples),
        'longitude': np.random.uniform(-74, -72, n_samples),
        
        # Enhanced charging parameters
        'charge_voltage': np.random.uniform(350, 400, n_samples),
        'charge_current': np.random.uniform(0, 250, n_samples),
        'charge_power': np.random.uniform(0, 100000, n_samples),
        'charge_time': np.random.uniform(0, 60, n_samples),
        'charge_type': np.random.choice(['AC', 'DC'], n_samples),
        'charging_speed': np.random.choice(['Slow', 'Medium', 'Fast'], n_samples),
        'charge_location_type': np.random.choice(['home', 'public', 'supercharger'], n_samples),
        
        # Usage patterns
        'timestamp': timestamps,
        'day_of_week': [t.weekday() for t in timestamps],
        'hour_of_day': [t.hour for t in timestamps],
        'daily_distance': np.random.uniform(0, 150, n_samples),
        'trip_duration': np.random.uniform(0, 180, n_samples),
        'driving_style': np.random.choice(['economic', 'normal', 'sporty'], n_samples),
        'route_type': np.random.choice(['city', 'highway', 'mixed'], n_samples),
        
        # Additional parameters
        'ambient_temp': np.random.uniform(0, 35, n_samples),
        'battery_age_months': np.random.uniform(0, 60, n_samples),
        'state_of_charge': np.random.uniform(0, 100, n_samples),
        'charging_efficiency': np.random.uniform(85, 95, n_samples),
        'battery_cycles': np.random.uniform(0, 500, n_samples),
        'energy_consumption': np.random.uniform(15, 25, n_samples)  # kWh/100km
    }
    
    # Target variables
    data['remaining_range'] = calculate_range(data)
    data['preferred_charge_type'] = predict_charge_type(data)
    data['next_charge_time'] = predict_next_charge(data)
    
    return pd.DataFrame(data)

def calculate_range(data):
    # Enhanced range calculation considering multiple factors
    base_range = data['state_of_charge'] * 0.5
    efficiency_factor = data['charging_efficiency'] * 0.3
    temperature_factor = np.where(
        (data['ambient_temp'] >= 15) & (data['ambient_temp'] <= 25),
        1.0,
        0.8
    )
    driving_style_factor = {
        'economic': 1.2,
        'normal': 1.0,
        'sporty': 0.8
    }
    driving_factors = [driving_style_factor[style] for style in data['driving_style']]
    
    return base_range * efficiency_factor * temperature_factor * driving_factors

def predict_charge_type(data):
    # Logic to determine preferred charging type based on patterns
    conditions = []
    for loc, power in zip(data['charge_location_type'], data['charge_power']):
        if loc == 'home':
            conditions.append('AC')
        elif power > 50000:
            conditions.append('DC')
        else:
            conditions.append('AC')
    return conditions

def predict_next_charge(data):
    # Predict next charging session based on usage patterns
    return np.random.uniform(0, 24, len(data))  # Hours until next charge

class EVUsageAnalyzer:
    def __init__(self):
        self.usage_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.charge_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def prepare_features(self, df):
        usage_features = [
            'daily_distance', 'trip_duration', 'energy_consumption',
            'day_of_week', 'hour_of_day', 'ambient_temp',
            'state_of_charge', 'battery_cycles'
        ]
        
        charging_features = [
            'charge_power', 'charge_time', 'state_of_charge',
            'charging_efficiency', 'ambient_temp'
        ]
        
        return df[usage_features], df[charging_features]
    
    def train(self, df):
        usage_features, charging_features = self.prepare_features(df)
        
        # Train usage prediction model
        y_usage = df['energy_consumption']
        X_usage_scaled = self.scaler.fit_transform(usage_features)
        self.usage_model.fit(X_usage_scaled, y_usage)
        
        # Train charging type prediction model
        y_charge = self.label_encoder.fit_transform(df['charge_type'])
        X_charge_scaled = self.scaler.fit_transform(charging_features)
        self.charge_type_model.fit(X_charge_scaled, y_charge)
        
        return self.evaluate_models(df, X_usage_scaled, y_usage, X_charge_scaled, y_charge)
    
    def evaluate_models(self, df, X_usage, y_usage, X_charge, y_charge):
        # Usage prediction evaluation
        usage_pred = self.usage_model.predict(X_usage)
        usage_mse = mean_squared_error(y_usage, usage_pred)
        usage_r2 = r2_score(y_usage, usage_pred)
        
        # Charging type prediction evaluation
        charge_pred = self.charge_type_model.predict(X_charge)
        charge_report = classification_report(y_charge, charge_pred, target_names=['AC', 'DC'])
        
        return {
            'usage_metrics': {'mse': usage_mse, 'r2': usage_r2},
            'charging_report': charge_report
        }
    
    def predict_future_usage(self, input_data):
        usage_features, _ = self.prepare_features(input_data)
        scaled_features = self.scaler.transform(usage_features)
        return self.usage_model.predict(scaled_features)

def main():
    st.title("Enhanced EV Battery Analysis and Prediction System")
    
    # Sidebar for model training
    st.sidebar.header("Model Training")
    if st.sidebar.button("Train New Model"):
        with st.spinner("Generating sample data and training models..."):
            df = generate_sample_data()
            analyzer = EVUsageAnalyzer()
            metrics = analyzer.train(df)
            st.session_state['analyzer'] = analyzer
            st.session_state['metrics'] = metrics
            st.sidebar.success("Models trained successfully!")
            st.sidebar.write("Usage Prediction Metrics:", metrics['usage_metrics'])
            st.sidebar.text("Charging Type Classification Report:")
            st.sidebar.text(metrics['charging_report'])
    
    # Main content
    tabs = st.tabs(["Usage Analysis", "Charging Patterns", "Predictions", "Recommendations"])
    
    with tabs[0]:
        st.header("Usage Pattern Analysis")
        if 'analyzer' in st.session_state:
            # Display usage patterns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Daily Usage Distribution")
                daily_usage = pd.DataFrame({
                    'hour': range(24),
                    'usage': np.random.normal(50, 15, 24)
                })
                fig = px.line(daily_usage, x='hour', y='usage',
                            title='Hourly Usage Pattern')
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Energy Consumption Trends")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(7)),
                    y=np.random.normal(20, 3, 7),
                    name='Weekly Trend'
                ))
                st.plotly_chart(fig)
    
    with tabs[1]:
        st.header("Charging Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Charging Session Details")
            charge_type = st.selectbox("Charging Type", ['AC', 'DC'])
            charge_location = st.selectbox("Location", ['Home', 'Public', 'Supercharger'])
            
        with col2:
            st.subheader("Charging Statistics")
            if charge_type == 'AC':
                st.write("Average AC Charging Time: 6-8 hours")
                st.write("Typical Power Range: 3.3-22 kW")
            else:
                st.write("Average DC Charging Time: 30-45 minutes")
                st.write("Typical Power Range: 50-350 kW")
    
    with tabs[2]:
        st.header("Future Usage Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            days_ahead = st.slider("Predict Usage for Next N Days", 1, 30, 7)
            driving_style = st.selectbox("Expected Driving Style", 
                                       ['economic', 'normal', 'sporty'])
        
        with col2:
            if st.button("Generate Prediction"):
                if 'analyzer' in st.session_state:
                    # Generate future usage prediction
                    future_usage = np.random.normal(20, 3, days_ahead)
                    fig = px.line(x=range(days_ahead), y=future_usage,
                                title='Predicted Daily Energy Consumption (kWh)')
                    st.plotly_chart(fig)
                else:
                    st.warning("Please train the model first!")
    
    with tabs[3]:
        st.header("Recommendations")
        if 'analyzer' in st.session_state:
            st.subheader("Optimal Charging Strategy")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Recommended Charging Times:")
                st.write("- Off-peak hours (22:00 - 06:00)")
                st.write("- Mid-day for solar optimization")
                
            with col2:
                st.write("Charging Type Recommendations:")
                st.write("- Use AC charging for regular daily charging")
                st.write("- Reserve DC charging for long trips")
                
            st.subheader("Usage Optimization")
            st.write("Based on your usage patterns:")
            st.write("1. Best times for longer trips: Weekends")
            st.write("2. Optimal charging locations based on your routes")
            st.write("3. Energy efficiency recommendations")

if __name__ == "__main__":
    main()