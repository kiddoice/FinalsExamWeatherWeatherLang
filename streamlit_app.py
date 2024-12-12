import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the saved model
@st.cache_resource
def load_model_from_file():
    model = load_model('weather_forecast_model2.h5')  # Replace with the path to your .h5 model file
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

# Make prediction on selected data
st.write("### Predict 'Rain Tomorrow'")
input_data = st.text_input("Enter the weather data for prediction (comma-separated):")

if input_data:
    try:
        # Convert the input into a list of floats
        input_data = [float(i) for i in input_data.split(',')]

        # Check if the input has exactly 5 features
        if len(input_data) != 5:
            raise ValueError("Please enter exactly 5 values corresponding to the features.")

        # Reshape the input data to the expected shape (1, 5)
        input_data = np.array(input_data).reshape(1, 5)

        # Assuming the model expects normalized data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)  # Scaling the input

        # Make prediction
        prediction = model.predict(input_data)

        # Get the probability of rain
        rain_probability = prediction[0][1]  # Index 1 corresponds to the "Rain" class

        # Display raw prediction for debugging
        st.write(f"Raw Prediction (Probability of Rain): {rain_probability:.2f}")

        # Adjust threshold if needed (e.g., using 0.5 for balanced classification)
        threshold = 0.5
        if rain_probability > threshold:
            rain_prediction = 'Rain'
            identifier = f"Rain (Probability: {rain_probability:.2f})"
        else:
            rain_prediction = 'No Rain'
            identifier = f"No Rain (Probability: {1 - rain_probability:.2f})"

        # Display the result
        st.write(f"Prediction: {rain_prediction}")
        st.write(f"Identifier: {identifier}")  # Show additional identifier info

    except Exception as e:
        st.error(f"Error in processing input data: {e}")
