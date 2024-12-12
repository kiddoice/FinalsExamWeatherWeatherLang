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
st.write(data[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']])

# Make prediction on selected data
st.write("### Predict 'Rain Tomorrow'")

# Input data for prediction (comma-separated)
input_data = st.text_input("Enter the weather data for prediction (comma-separated):")

# Show example No Rain and Rain values
st.write("### Example Values")

st.write("**Example values for 'No Rain':**")
st.write("[12.0, 22.0, 0.0, 60.0, 55.0]")  # No rain example

st.write("**Example values for 'Rain':**")
st.write("[15.0, 18.0, 5.0, 90.0, 85.0]")  # Rain example

# Adding a horizontal division (line) between the example values and input field
st.markdown("<hr>", unsafe_allow_html=True)

# Output prediction after user input
if input_data:
    try:
        # Convert the input into a list of floats
        input_data = [float(i) for i in input_data.split(',')]

        # Check if the input has exactly 5 features
        if len(input_data) != 5:
            raise ValueError("Please enter exactly 5 values corresponding to the features.")

        # Ensure the input data is normalized like the training data
        # Use the loaded scaler to transform the input data
        input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

        # Reshape the input data to match the model's expected input shape
        input_data_reshaped = input_data_scaled.reshape(1, 5)  # For a fully connected network (not CNN)

        # Make prediction
        prediction = model.predict(input_data_reshaped)

        # Debug: Display raw prediction (probabilities for both classes)
        st.write(f"Raw prediction (probabilities): {prediction[0]}")  # Shows the probability for both classes

        # Extract the scalar prediction (probability) and display it
        rain_probability = prediction[0][1]  # Probability of rain (index 1 for rain class)

        # Display raw prediction for debugging
        st.write(f"Raw Prediction (Probability of Rain): {rain_probability:.2f}")

        # Adjust threshold if needed (e.g., using 0.5 for a balanced classification)
        threshold = 0.5  # Try using a more neutral threshold
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
