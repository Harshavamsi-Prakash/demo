# Corrected app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import time

# Configuration
st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
st.title("🌪️ Advanced Wind Speed Prediction")

# App description
st.markdown("""
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: #004080;">About This App</h3>
    <p>This application predicts wind speed with high accuracy using multiple machine learning models trained on real-time weather data.</p>
</div>
""", unsafe_allow_html=True)

def get_coordinates(city_name):
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
    headers = {"User-Agent": "WindSpeedPredictor/1.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        location_data = response.json()
        if location_data:
            location = location_data[0]
            return float(location['lat']), float(location['lon'])
        st.warning("City not found. Please check the spelling.")
        return None, None
    except Exception as e:
        st.error(f"Geocoding failed: {str(e)}")
        return None, None

def get_weather_data(lat, lon, hours):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,pressure_msl,precipitation,cloudcover&windspeed_unit=ms&forecast_days=2"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            "time": [datetime.now() + timedelta(hours=i) for i in range(hours)],
            "temperature": data['hourly']['temperature_2m'][:hours],
            "humidity": data['hourly']['relativehumidity_2m'][:hours],
            "pressure": data['hourly']['pressure_msl'][:hours],
            "precipitation": data['hourly']['precipitation'][:hours],
            "cloud_cover": data['hourly']['cloudcover'][:hours],
            "actual_wind_speed": data['hourly']['windspeed_10m'][:hours]
        })
        
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        return df.dropna()
    except Exception as e:
        st.error(f"Weather data fetch failed: {str(e)}")
        return None

def train_models(df, selected_models):
    results = []
    
    X = df[['temperature', 'humidity', 'pressure', 'precipitation', 
            'cloud_cover', 'hour', 'day_of_week', 'is_daytime']]
    y = df['actual_wind_speed']
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
        "Neural Network": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42))
    }
    
    for model_name in selected_models:
        start_time = time.time()
        model = models[model_name]
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = max(0, min(100, 100 * (1 - mse/np.var(y_test))))
        
        results.append({
            "name": model_name,
            "model": model,
            "mse": mse,
            "r2": r2,
            "accuracy": accuracy,
            "training_time": time.time() - start_time
        })
    
    return results, X_test, y_test

# UI Elements
col1, col2 = st.columns(2)
with col1:
    city_name = st.text_input("Enter City Name", "London")
with col2:
    forecast_hours = st.slider("Forecast Hours", 12, 48, 24, 12)

selected_models = st.multiselect(
    "Select ML Models",
    options=["Random Forest", "Gradient Boosting", "Support Vector Machine", "Neural Network"],
    default=["Random Forest", "Gradient Boosting"]
)

if st.button("Predict Wind Speed", type="primary"):
    with st.spinner("Fetching weather data and training models..."):
        lat, lon = get_coordinates(city_name)
        if lat is None:
            st.stop()
        
        weather_df = get_weather_data(lat, lon, forecast_hours)
        if weather_df is None:
            st.stop()
        
        st.subheader(f"Current Weather in {city_name}")
        current = weather_df.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Temperature", f"{current['temperature']:.1f}°C")
        cols[1].metric("Humidity", f"{current['humidity']:.0f}%")
        cols[2].metric("Pressure", f"{current['pressure']:.0f} hPa")
        cols[3].metric("Wind Speed", f"{current['actual_wind_speed']:.1f} m/s")
        cols[4].metric("Cloud Cover", f"{current['cloud_cover']:.0f}%")
        
        model_results, X_test, y_test = train_models(weather_df, selected_models)
        
        for result in model_results:
            weather_df[f"predicted_{result['name']}"] = result["model"].predict(weather_df[X_test.columns])
        
        # Fixed the dataframe display syntax
        perf_df = pd.DataFrame([{
            "Model": r["name"],
            "Accuracy (%)": r["accuracy"],
            "R² Score": r["r2"],
            "Training Time (s)": r["training_time"]
        } for r in model_results])
        
        st.dataframe(perf_df.style.format({
            "Accuracy (%)": "{:.1f}",
            "R² Score": "{:.3f}",
            "Training Time (s)": "{:.2f}"
        }))
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=weather_df["time"],
            y=weather_df["actual_wind_speed"],
            name="Actual Wind Speed",
            line=dict(color='blue', width=2)
        ))
        
        for result in model_results:
            fig1.add_trace(go.Scatter(
                x=weather_df["time"],
                y=weather_df[f"predicted_{result['name']}"],
                name=f"{result['name']} Prediction",
                line=dict(dash='dash')
            ))
        
        fig1.update_layout(
            xaxis_title="Time",
            yaxis_title="Wind Speed (m/s)",
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)

# requirements.txt without versions
