import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib  # To load the scaler.pkl

# Load the saved model
@st.cache_resource
def load_model_from_file():
    model = load_model('weather_forecast_model4.h5')  # Replace with the path to your .h5 model file
    return model

# Load the scaler
@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.pkl')  # Replace with the path to your scaler.pkl file
    return scaler

# Load the Weather Dataset
@st.cache
def load_data():
    # Replace 'weather.csv' with the actual path or raw GitHub URL if hosted online
    data = pd.read_csv('weather.csv')
    return data

# Create a Streamlit app
st.title("🌦️ Weather Dataset Explorer")
st.write("Explore weather conditions and forecasts.")

# Load the data
data = load_data()

# Load the model
model = load_model_from_file()

# Load the scaler
scaler = load_scaler()

# Display the data
st.write("### Weather Dataset")
st.write(data)

# Filter by Rain Tomorrow
st.write("### Filter by 'Rain Tomorrow'")
rain_tomorrow_filter = st.selectbox("Will it rain tomorrow?", options=data['RainTomorrow'].unique())
filtered_data = data[data['RainTomorrow'] == rain_tomorrow_filter]
st.write("### Filtered Data")
st.write(filtered_data)

# Filter by Temperature Range
st.write("### Filter by Temperature Range")
min_temp = st.slider("Minimum Temperature", int(data['MinTemp'].min()), int(data['MinTemp'].max()))
max_temp = st.slider("Maximum Temperature", int(data['MaxTemp'].min()), int(data['MaxTemp'].max()))
temp_filtered_data = data[(data['MinTemp'] >= min_temp) & (data['MaxTemp'] <= max_temp)]
st.write("### Filtered Data by Temperature Range")
st.write(temp_filtered_data)

# Display key metrics
st.write("### Weather Metrics Overview")
st
