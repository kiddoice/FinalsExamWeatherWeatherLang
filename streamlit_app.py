# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the saved model
@st.cache_resource
def load_model_from_file():
    model = load_model('weather_forecast_model.h5')  # Replace with the path to your .h5 model file
    return model

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
st.write(data[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']])

# Make prediction on selected data
st.write("### Predict 'Rain Tomorrow'")
input_data = st.text_input("Enter the weather data for prediction (comma-separated):")

if input_data:
    try:
        # Preprocessing the input data
        input_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, 5, 1, 1)
        
        # Assuming the model expects normalized data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data.reshape(1, -1))  # Flatten for scaling then reshape
        
        # Make prediction
        prediction = model.predict(input_data)
        rain_prediction = 'Rain' if prediction[0] > 0.5 else 'No Rain'
        st.write(f"Prediction: {rain_prediction}")
    except Exception as e:
        st.error(f"Error in processing input data: {e}")

# Display additional insights
st.write("### Additional Weather Insights")
if 'Rainfall' in data.columns:
    avg_rainfall = data['Rainfall'].mean()
    st.write(f"**Average Rainfall**: {avg_rainfall:.2f} mm")
