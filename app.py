import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
import json
from datetime import datetime, timedelta
import time
import pickle

# Title of the app
st.title("🌬️ Real-time Wind Speed Prediction with OpenWeatherMap API")

# Sidebar configuration
st.sidebar.header("API & Model Configuration")
API_KEY = st.sidebar.text_input("OpenWeatherMap API Key", "e0100edeedd99f5ae298581c486626a4")
city_name = st.sidebar.text_input("City Name", "London")
country_code = st.sidebar.text_input("Country Code (optional)", "GB")

# Function to get coordinates from OpenWeatherMap
def get_city_coordinates(city, country=None, api_key=API_KEY):
    base_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": f"{city},{country}" if country else city,
        "limit": 1,
        "appid": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            st.error("City not found. Please check the name and try again.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, None

# Function to fetch weather data
def fetch_weather_data(lat, lon, api_key, days=5):
    base_url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "minutely,daily,alerts",
        "units": "metric",
        "appid": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process hourly data
        hourly_data = data.get('hourly', [])
        records = []
        for hour in hourly_data:
            dt = datetime.fromtimestamp(hour['dt'])
            records.append({
                "datetime": dt,
                "temp": hour['temp'],
                "pressure": hour['pressure'],
                "humidity": hour['humidity'],
                "wind_speed": hour['wind_speed'],
                "wind_deg": hour['wind_deg'],
                "clouds": hour['clouds']
            })
        
        return pd.DataFrame(records)
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch weather data: {e}")
        return None

# Function to train/evaluate model
def train_model(df):
    # Feature engineering
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Features and target
    features = ['temp', 'pressure', 'humidity', 'wind_deg', 'clouds', 'hour', 'day_of_week', 'month']
    X = df[features]
    y = df['wind_speed']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_test, y_test, y_pred

# Main app logic
if st.sidebar.button("Fetch Data & Train Model"):
    with st.spinner("Fetching weather data..."):
        lat, lon = get_city_coordinates(city_name, country_code if country_code else None)
        
        if lat and lon:
            st.success(f"Coordinates found: Latitude {lat}, Longitude {lon}")
            weather_df = fetch_weather_data(lat, lon, API_KEY)
            
            if weather_df is not None:
                st.session_state.weather_df = weather_df
                
                # Show raw data
                st.subheader("Raw Weather Data")
                st.dataframe(weather_df.head())
                
                # Train model
                with st.spinner("Training model..."):
                    model, mse, r2, X_test, y_test, y_pred = train_model(weather_df)
                    st.session_state.model = model
                    
                    # Display metrics
                    st.subheader("Model Evaluation")
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Squared Error", f"{mse:.2f}")
                    col2.metric("R² Score", f"{r2:.2f}")
                    
                    # Plot actual vs predicted
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        labels={'x': 'Actual Wind Speed (m/s)', 'y': 'Predicted Wind Speed (m/s)'},
                        title="Actual vs Predicted Wind Speed"
                    )
                    fig.add_shape(
                        type="line",
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max(),
                        line=dict(color="Red", dash="dash")
                    )
                    st.plotly_chart(fig)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': X_test.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig2 = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance'
                    )
                    st.plotly_chart(fig2)

# Prediction section
if 'model' in st.session_state and 'weather_df' in st.session_state:
    st.subheader("Make Predictions")
    
    # Use last available data point for prediction
    last_data = st.session_state.weather_df.iloc[-1].copy()
    
    # Create input form with current values as defaults
    with st.form("prediction_form"):
        st.write("Adjust parameters for prediction:")
        
        col1, col2 = st.columns(2)
        temp = col1.number_input("Temperature (°C)", value=last_data['temp'])
        pressure = col2.number_input("Pressure (hPa)", value=last_data['pressure'])
        
        col3, col4 = st.columns(2)
        humidity = col3.number_input("Humidity (%)", value=last_data['humidity'])
        wind_deg = col4.number_input("Wind Direction (degrees)", value=last_data['wind_deg'])
        
        col5, col6 = st.columns(2)
        clouds = col5.number_input("Cloud Coverage (%)", value=last_data['clouds'])
        hour = col6.number_input("Hour of Day", min_value=0, max_value=23, value=last_data['hour'])
        
        submitted = st.form_submit_button("Predict Wind Speed")
        
        if submitted:
            # Prepare input features
            input_data = pd.DataFrame({
                'temp': [temp],
                'pressure': [pressure],
                'humidity': [humidity],
                'wind_deg': [wind_deg],
                'clouds': [clouds],
                'hour': [hour],
                'day_of_week': [last_data['day_of_week']],
                'month': [last_data['month']]
            })
            
            # Make prediction
            prediction = st.session_state.model.predict(input_data)[0]
            actual = last_data['wind_speed']
            
            # Display results
            st.metric("Predicted Wind Speed", f"{prediction:.2f} m/s")
            st.metric("Actual Wind Speed (from API)", f"{actual:.2f} m/s")
            
            # Show error
            error = abs(prediction - actual)
            st.metric("Absolute Error", f"{error:.2f} m/s")

# Time series visualization
if 'weather_df' in st.session_state:
    st.subheader("Wind Speed Time Series")
    
    # Plot wind speed over time
    fig3 = px.line(
        st.session_state.weather_df,
        x='datetime',
        y='wind_speed',
        title='Wind Speed Over Time'
    )
    st.plotly_chart(fig3)
    
    # Show correlation matrix
    st.subheader("Feature Correlations")
    numeric_df = st.session_state.weather_df.select_dtypes(include=[np.number])
    fig4 = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        title="Correlation Matrix"
    )
    st.plotly_chart(fig4)
  
