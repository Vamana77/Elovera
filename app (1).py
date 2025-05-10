import streamlit as st
import pandas as pd
import joblib

# Step 1: Load the model and feature names
model, expected_columns = joblib.load("model_with_features.pkl")

st.title("Prediction App")

# Step 2: Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=10000, value=50000)
score = st.number_input("Score", min_value=0.0, max_value=1.0, value=0.8)

# Step 3: Create a DataFrame from user input
input_data = {
    "age": age,
    "income": income,
    "score": score
}
input_df = pd.DataFrame([input_data])

# ✅ Step 4: Reorder columns to match training data
input_df = input_df[expected_columns]

# Step 5: Make prediction
prediction = model.predict(input_df)[0]

# Step 6: Display result
st.success(f"Prediction: {prediction}")
