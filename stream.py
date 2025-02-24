import streamlit as st
import joblib  # To load the .pkl model
import numpy as np

# Load the trained model
model = joblib.load("dib_model.pkl")  # Ensure model.pkl is in the same directory

# Streamlit App Title
st.title("Machine Learning Model Deployment with Streamlit")
st.write("Enter the required input values and get a prediction from the trained model.")

# Define the input fields
st.sidebar.header("User Input")

# Example: Assuming the model expects 3 numerical inputs
feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
feature3 = st.sidebar.number_input("Feature 4", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
feature4 = st.sidebar.number_input("Feature 5", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
feature5 = st.sidebar.number_input("Feature 6", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
feature6 = st.sidebar.number_input("Feature 7", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
feature7 = st.sidebar.number_input("Feature 8", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
feature8 = st.sidebar.number_input("Feature 9", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
# Collect input as a NumPy array
input_data = np.array([[feature1, feature2, feature3,feature4,feature5,feature6,feature7,feature8]])

# Predict button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")

# Run this script with: streamlit run script_name.py
