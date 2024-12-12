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
        # Convert the input into a list of floats
        input_data = [float(i) for i in input_data.split(',')]

        # Check if the input has exactly 5 features
        if len(input_data) != 5:
            raise ValueError("Please enter exactly 5 values corresponding to the features.")

        # Reshape the input data to the expected shape (1, 5, 1, 1)
        input_data = np.array(input_data).reshape(1, 5, 1, 1)

        # Assuming the model expects normalized data
        # Flatten for scaling and then reshape it back to (1, 5, 1, 1)
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data.reshape(1, -1))  # Flatten for scaling
        input_data = input_data.reshape(1, 5, 1, 1)  # Reshape back to (1, 5, 1, 1)

        # Make prediction
        prediction = model.predict(input_data)

        # Extract the scalar prediction (probability) and display it
        rain_probability = prediction[0][0]  # Probability of rain (0 - 1)

        # Display raw prediction for debugging
        st.write(f"Raw Prediction (Probability of Rain): {rain_probability:.2f}")

        # Adjust threshold if needed (e.g., using 0.6 or 0.7 for a stricter classification)
        threshold = 0.3
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



# Display additional insights
st.write("### Additional Weather Insights")
if 'Rainfall' in data.columns:
    avg_rainfall = data['Rainfall'].mean()
    st.write(f"**Average Rainfall**: {avg_rainfall:.2f} mm")
