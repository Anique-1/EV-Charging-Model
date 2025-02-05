import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load and preprocess data
def preprocess_data(df):
    # Convert datetime columns
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])
    
    # Extract time features
    df['Hour'] = df['Charging Start Time'].dt.hour
    df['Month'] = df['Charging Start Time'].dt.month
    df['Day'] = df['Charging Start Time'].dt.day
    
    # Handle missing values
    df['Distance Driven (since last charge) (km)'].fillna(df['Distance Driven (since last charge) (km)'].mean(), inplace=True)
    df['Charging Rate (kW)'].fillna(df['Charging Rate (kW)'].mean(), inplace=True)
    df['Energy Consumed (kWh)'].fillna(df['Energy Consumed (kWh)'].mean(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['Vehicle Model', 'Charging Station Location', 'Time of Day', 
                         'Day of Week', 'Charger Type', 'User Type']
    
    for col in categorical_columns:
        df[col + '_encoded'] = le.fit_transform(df[col])
    
    return df

def prepare_features(df):
    features = ['Battery Capacity (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
                'Energy Consumed (kWh)', 'Temperature (°C)', 'Vehicle Age (years)',
                'State of Charge (Start %)', 'State of Charge (End %)', 'Hour', 'Month', 'Day',
                'Vehicle Model_encoded', 'Charging Station Location_encoded', 'Time of Day_encoded',
                'Day of Week_encoded', 'Charger Type_encoded', 'User Type_encoded']
    
    return df[features]

# Load the data
df = pd.read_csv('ev_charging_patterns.csv')
df = preprocess_data(df)

# Prepare features and targets
X = prepare_features(df)
y_cost = df['Charging Cost (USD)']
y_distance = df['Distance Driven (since last charge) (km)']

# Split the data
X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
_, _, y_distance_train, y_distance_test = train_test_split(X, y_distance, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
distance_model = RandomForestRegressor(n_estimators=100, random_state=42)

cost_model.fit(X_train_scaled, y_cost_train)
distance_model.fit(X_train_scaled, y_distance_train)

# Calculate feature importance
cost_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': cost_model.feature_importances_
}).sort_values('importance', ascending=False)

distance_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': distance_model.feature_importances_
}).sort_values('importance', ascending=False)

# Save models and scaler
with open('cost_model.pkl', 'wb') as f:
    pickle.dump(cost_model, f)
    
with open('distance_model.pkl', 'wb') as f:
    pickle.dump(distance_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model Performance:")
print(f"Cost Model R2 Score: {cost_model.score(X_test_scaled, y_cost_test):.3f}")
print(f"Distance Model R2 Score: {distance_model.score(X_test_scaled, y_distance_test):.3f}")

print("\nTop 5 Important Features for Cost Prediction:")
print(cost_importance.head())

print("\nTop 5 Important Features for Distance Prediction:")
print(distance_importance.head())