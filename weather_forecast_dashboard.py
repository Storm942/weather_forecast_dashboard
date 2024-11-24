import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model and scaler
model = load_model(r'C:\jupyter notebook files\weather_forecast_model_new.keras')
scaler = joblib.load(r'C:\jupyter notebook files\scaler.pkl')

# Set up Streamlit page
st.set_page_config(page_title="Weather Forecasting Dashboard", layout="wide")
st.title("Weather Forecasting Dashboard")

# Sidebar for user input
st.sidebar.header("Input Weather Data")
temperature_input = [st.sidebar.number_input(f"Temperature (°C) - Day {i+1}", value=20.0) for i in range(7)]
humidity_input = [st.sidebar.number_input(f"Humidity (%) - Day {i+1}", value=60.0) for i in range(7)]
precipitation_input = [st.sidebar.number_input(f"Precipitation (mm) - Day {i+1}", value=0.0) for i in range(7)]
wind_speed_input = [st.sidebar.number_input(f"Wind Speed (km/h) - Day {i+1}", value=10.0) for i in range(7)]

# Create input DataFrame
input_data = pd.DataFrame({
    'Temperature_C': temperature_input,
    'Humidity_pct': humidity_input,
    'Precipitation_mm': precipitation_input,
    'Wind_Speed_kmh': wind_speed_input
})

# Scale input data and reshape for LSTM
scaled_input_data = scaler.transform(input_data)
input_seq = scaled_input_data.reshape(1, 7, len(input_data.columns))

# Predict the next day's temperature
predicted_temp = model.predict(input_seq)

# Combine predicted temperature with dummy features
# Assuming the scaler was fitted on 4 features, we need to add 3 dummy features
dummy_features = np.zeros((predicted_temp.shape[0], 3))  # 3 dummy features for Humidity, Precipitation, Wind Speed
predicted_temp_combined = np.hstack((predicted_temp, dummy_features))

# Perform inverse transformation and extract the predicted temperature
predicted_temp_rescaled = scaler.inverse_transform(predicted_temp_combined)[:, 0]

# Display predicted temperature
st.subheader(f"Predicted Temperature for the Next Day: {predicted_temp_rescaled[0]:.2f} °C")

# Plot input and predicted temperature
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 8), temperature_input, label="Input Temperature", marker='o', color='blue')
ax.plot(8, predicted_temp_rescaled[0], label="Predicted Temperature", marker='o', color='red')
ax.set_title("Temperature Forecast vs Input Data")
ax.set_xlabel("Days")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

# Multi-day forecast
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=14, value=7)

def forecast_multiple_days(model, input_seq, scaler, forecast_horizon):
    predictions = []
    for _ in range(forecast_horizon):
        pred = model.predict(input_seq)
        predictions.append(pred[0, 0])
        
        # Update input_seq with the latest prediction
        new_seq = np.roll(input_seq[0], -1, axis=0)
        new_seq[-1, 0] = pred[0, 0]  # Replace the oldest temperature with the new prediction
        input_seq = new_seq.reshape(1, 7, len(input_seq[0, 0]))

    # Rescale predictions to original values
    dummy_features = np.zeros((len(predictions), 3))  # 3 dummy features for Humidity, Precipitation, Wind Speed
    predictions_combined = np.hstack((np.array(predictions).reshape(-1, 1), dummy_features))
    predictions_rescaled = scaler.inverse_transform(predictions_combined)[:, 0]
    return predictions_rescaled

if forecast_days > 1:
    future_predictions = forecast_multiple_days(model, input_seq, scaler, forecast_horizon=forecast_days)
    st.subheader(f"Temperature Forecast for the Next {forecast_days} Days")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, forecast_days + 1), future_predictions, label="Future Predictions", marker='o', color='red')
    ax.set_title(f"Temperature Forecast for the Next {forecast_days} Days")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Temperature ")