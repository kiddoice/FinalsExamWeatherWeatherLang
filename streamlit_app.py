import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

@st.cache_resource
def load_model_from_file():
    model = load_model('weather_forecast_model4.h5')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.pkl')
    return scaler

@st.cache
def load_data():
    data = pd.read_csv('weather.csv')
    return data

st.title("🌦️ Weather Dataset Explorer")
st.write("Explore weather conditions and forecasts.")

data = load_data()

model = load_model_from_file()

scaler = load_scaler()

st.write("### Weather Dataset")
st.write(data)

st.write("### Filter by 'Rain Tomorrow'")
rain_tomorrow_filter = st.selectbox("Will it rain tomorrow?", options=data['RainTomorrow'].unique())
filtered_data = data[data['RainTomorrow'] == rain_tomorrow_filter]
st.write("### Filtered Data")
st.write(filtered_data)

st.write("### Filter by Temperature Range")
min_temp = st.slider("Minimum Temperature", int(data['MinTemp'].min()), int(data['MinTemp'].max()))
max_temp = st.slider("Maximum Temperature", int(data['MaxTemp'].min()), int(data['MaxTemp'].max()))
temp_filtered_data = data[(data['MinTemp'] >= min_temp) & (data['MaxTemp'] <= max_temp)]
st.write("### Filtered Data by Temperature Range")
st.write(temp_filtered_data)

st.write("### Weather Metrics Overview")
st.write(data[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']])

st.write("### Predict 'Rain Tomorrow'")

input_data = st.text_input("Enter the weather data for prediction (comma-separated):")

st.write("### Example Values")

st.write("**Example values for 'No Rain':**")
st.write("[12.0, 22.0, 0.0, 60.0, 55.0]")

st.write("**Example values for 'Rain':**")
st.write("[15.0, 18.0, 5.0, 90.0, 85.0]")

st.markdown("<hr>", unsafe_allow_html=True)

if input_data:
    try:
        input_data = [float(i) for i in input_data.split(',')]

        if len(input_data) != 5:
            raise ValueError("Please enter exactly 5 values corresponding to the features.")

        # Calculate the TempDiff feature (MaxTemp - MinTemp)
        temp_diff = input_data[1] - input_data[0]  # MaxTemp - MinTemp
        input_data.append(temp_diff)  # Add TempDiff to the input data

        # Ensure the input data is scaled with the updated scaler
        input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

        input_data_reshaped = input_data_scaled.reshape(1, 6)  # Now there are 6 features

        prediction = model.predict(input_data_reshaped)

        st.write(f"Raw prediction (probabilities): {prediction[0]}")

        rain_probability = prediction[0][1]

        st.write(f"Raw Prediction (Probability of Rain): {rain_probability:.2f}")

        threshold = 0.7
        if rain_probability > threshold:
            rain_prediction = 'Rain'
            identifier = f"Rain (Probability: {rain_probability:.2f})"
        else:
            rain_prediction = 'No Rain'
            identifier = f"No Rain (Probability: {1 - rain_probability:.2f})"

        st.write(f"Prediction: {rain_prediction}")
        st.write(f"Identifier: {identifier}")

    except Exception as e:
        st.error(f"Error in processing input data: {e}")

st.write("### Additional Weather Insights")
if 'Rainfall' in data.columns:
    avg_rainfall = data['Rainfall'].mean()
    st.write(f"**Average Rainfall**: {avg_rainfall:.2f} mm")
