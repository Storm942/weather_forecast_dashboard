import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model and scaler
model = load_model('C:/jupyter notebook files/weather_forecast_model_new.keras')
scaler = joblib.load('C:/jupyter notebook files/scaler.pkl')

# Set up page configuration
st.set_page_config(page_title="Weather Forecasting Dashboard", layout="wide")
st.title("Weather Forecasting Dashboard")

# User Input
st.sidebar.header("Input Weather Data")
temperature_input = [st.sidebar.number_input(f"Temperature (°C) - Day {i+1}", value=20.0) for i in range(7)]
humidity_input = [st.sidebar.number_input(f"Humidity (%) - Day {i+1}", value=60.0) for i in range(7)]
precipitation_input = [st.sidebar.number_input(f"Precipitation (mm) - Day {i+1}", value=0.0) for i in range(7)]
wind_speed_input = [st.sidebar.number_input(f"Wind Speed (km/h) - Day {i+1}", value=10.0) for i in range(7)]

# Create DataFrame
input_data = pd.DataFrame({
    'Temperature_C': temperature_input,
    'Humidity_pct': humidity_input,
    'Precipitation_mm': precipitation_input,
    'Wind_Speed_kmh': wind_speed_input
})

# Scale the input data
scaled_input_data = scaler.transform(input_data)
input_seq = scaled_input_data.reshape(1, 7, len(input_data.columns))

# Make a prediction
predicted_temp = model.predict(input_seq)

# Debugging logs
st.text(f"Shape of predicted_temp: {predicted_temp.shape}")
st.text(f"Scaler expects {len(scaler.min_)} features.")
st.text(f"Input data columns: {input_data.columns.tolist()}")

# Select the first column (temperature prediction) and reshape
predicted_temp_trimmed = predicted_temp[:, :1]  # Assuming the first column is temperature
st.text(f"Shape of predicted_temp_trimmed: {predicted_temp_trimmed.shape}")

# Add dummy features to match scaler's expected shape
dummy_features = np.zeros((predicted_temp_trimmed.shape[0], len(input_data.columns) - 1))
predicted_temp_combined = np.c_[predicted_temp_trimmed, dummy_features]

# Debug combined shape
st.text(f"Shape of predicted_temp_combined: {predicted_temp_combined.shape}")

# Initialize predicted_temp_rescaled for fallback
predicted_temp_rescaled = None

# Perform inverse transformation
try:
    predicted_temp_rescaled = scaler.inverse_transform(predicted_temp_combined)[:, 0]
except ValueError as e:
    st.error(f"Error during inverse transformation: {e}")

# Fallback if inverse transform fails
if predicted_temp_rescaled is None:
    st.error("Unable to display the predicted temperature due to a processing error.")
else:
    # Display the predicted temperature
    st.subheader(f"Predicted Temperature for the Next Day: {predicted_temp_rescaled[0]:.2f} °C")

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 8), temperature_input, label="Input Temperature", marker='o', color='blue')
    ax.plot(8, predicted_temp_rescaled[0], label="Predicted Temperature", marker='o', color='red')
    ax.set_title("Temperature Forecast vs Input Data")
    ax.set_xlabel("Days")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)
