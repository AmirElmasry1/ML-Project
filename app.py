import streamlit as st
import joblib
import numpy as np

# Load the trained Decision Tree model
model = joblib.load("best_decision_tree_model.pkl")

# Streamlit UI
st.title("Air Quality Prediction App")
st.write("Enter the input features to predict the Air Quality level:")

# Example input fields (replace with actual feature names from your dataset)
feature1 = st.number_input("Feature 1 (e.g., Temperature)", value=0.0)
feature2 = st.number_input("Feature 2 (e.g., Humidity)", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Add all features as a NumPy array
input_features = np.array([[feature1, feature2, feature3, feature4]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.write(f"Predicted Air Quality: {prediction[0]}")
