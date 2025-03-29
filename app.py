import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# App title
st.title("🌬️ Free Wind Speed Predictor")

# API Configuration
API_KEY = "fca4013c19fa681ae299131658ea2dac"  # Your free-tier key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"  # Free endpoint

# Sidebar inputs
st.sidebar.header("Configuration")
city_name = st.sidebar.text_input("City Name", "London")
country_code = st.sidebar.text_input("Country Code (optional)", "")

def get_weather_data():
    """Fetch current weather data using free API endpoint"""
    params = {
        "q": f"{city_name},{country_code}" if country_code else city_name,
        "units": "metric",
        "appid": API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('cod') != 200:
            st.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            "temp": data['main']['temp'],
            "pressure": data['main']['pressure'],
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed'],
            "wind_deg": data['wind'].get('deg', 0),
            "clouds": data['clouds'].get('all', 0),
            "timestamp": datetime.fromtimestamp(data['dt'])
        }
        
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

# Generate synthetic training data (since historical API isn't free)
@st.cache_data
def generate_training_data():
    np.random.seed(42)
    days = 30
    hours = days * 24
    
    base = 5 + 5 * np.sin(2 * np.pi * np.arange(hours) / 24)
    wind_speed = base + np.random.normal(0, 2, hours)
    
    df = pd.DataFrame({
        "temp": 10 + 10 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.random.normal(0, 3, hours),
        "pressure": 1010 + np.random.normal(0, 5, hours),
        "humidity": 50 + 30 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.random.normal(0, 10, hours),
        "wind_deg": np.random.randint(0, 360, hours),
        "clouds": np.random.randint(0, 100, hours),
        "hour": np.tile(np.arange(24), days),
        "wind_speed": wind_speed
    })
    
    return df

# Train model
def train_wind_model(df):
    features = ['temp', 'pressure', 'humidity', 'wind_deg', 'clouds', 'hour']
    X = df[features]
    y = df['wind_speed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Main app
if st.button("Get Current Wind Data"):
    with st.spinner("Fetching weather data..."):
        weather_data = get_weather_data()
        
        if weather_data:
            st.session_state.current_data = weather_data
            st.success(f"Data fetched for {city_name} at {weather_data['timestamp']}")
            
            # Display current conditions
            cols = st.columns(4)
            cols[0].metric("Temperature", f"{weather_data['temp']}°C")
            cols[1].metric("Pressure", f"{weather_data['pressure']} hPa")
            cols[2].metric("Humidity", f"{weather_data['humidity']}%")
            cols[3].metric("Wind Speed", f"{weather_data['wind_speed']} m/s")
            
            # Generate and train on synthetic data
            training_df = generate_training_data()
            model, X_test, y_test = train_wind_model(training_df)
            
            # Prepare current data for prediction
            current_hour = weather_data['timestamp'].hour
            input_data = pd.DataFrame([{
                'temp': weather_data['temp'],
                'pressure': weather_data['pressure'],
                'humidity': weather_data['humidity'],
                'wind_deg': weather_data['wind_deg'],
                'clouds': weather_data['clouds'],
                'hour': current_hour
            }])
            
            # Make prediction
            predicted_speed = model.predict(input_data)[0]
            actual_speed = weather_data['wind_speed']
            
            # Show results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Wind Speed", f"{predicted_speed:.2f} m/s")
            col2.metric("Actual Wind Speed", f"{actual_speed:.2f} m/s")
            
            # Show error
            error = abs(predicted_speed - actual_speed)
            st.metric("Absolute Error", f"{error:.2f} m/s")
            
            # Feature importance
            st.subheader("Model Insights")
            importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig)

# Add disclaimer
st.sidebar.markdown("""
**Note:** This free version uses:
- Current weather API (no historical data)
- Synthetic training data
- Limited prediction accuracy
""")
