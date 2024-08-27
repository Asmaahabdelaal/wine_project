import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import euclidean_distances

# Load the KNN model from the same directory
knn_model = joblib.load("model.pkl")

# Streamlit app title
st.title("Wine Type Prediction - Epsilon AI Final Project")

# Description
st.image("static/wine.jfif", width=300)
st.write("""
    This project predicts the type of wine based on its chemical properties.
    It's useful for vintners to classify wine quality, helping in better quality control.
""")

# Input fields
volatile_acidity = st.number_input('Volatile Acidity (0.1-1.5):', min_value=0.1, max_value=1.5, step=0.01)
chlorides = st.number_input('Chlorides (0.01-0.1):', min_value=0.01, max_value=0.1, step=0.01)

# Predict button
if st.button("Predict"):
    # Prepare input for prediction
    features = np.array([[volatile_acidity, chlorides]])

    # Predict with the KNN model
    prediction = knn_model.predict(features)[0]

    # Map numerical prediction to wine types
    wine_types = {0: 'White Wine', 1: 'Red Wine'}  # Adjust according to your mapping
    prediction_text = f"Predicted Wine Type: {wine_types.get(prediction, 'Unknown')}"

    # Display prediction
    st.success(prediction_text)
