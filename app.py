# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("compressed_log_transformed_model.pkl")  # Update path if needed
mapping_set = joblib.load("encoders.pkl")  # Save this from your notebook

# Define input fields
st.title("JEE College Closing Rank Predictor")

st.markdown("Enter the details below to predict the expected closing rank for your preferred branch:")

# Form inputs
year = st.number_input("Enter Year", min_value=2016, max_value=2050, step=1)
round_no = st.selectbox("Round Number", [1, 2, 3, 4, 5, 6, 7])
quota = st.selectbox("Quota", ["AI", "HS", "OS"])
gender = st.selectbox("Gender", ["Both", "Female"])
institute_name = st.selectbox("Institute", sorted(mapping_set['institute_name'].classes_))
branch = st.selectbox("Branch", sorted(mapping_set['branch'].classes_))
category = st.selectbox("Category", sorted(mapping_set['category'].classes_))

# Submit button
if st.button("Predict Closing Rank"):
    # Prepare input DataFrame
    input_data = pd.DataFrame({
        'year': [year],
        'round_no': [round_no],
        'quota': [quota],
        'gender': [gender],
        'institute_name': [institute_name],
        'branch': [branch],
        'category': [category]
    })

    # Encode using saved LabelEncoders
    for col in input_data.columns:
        if col in mapping_set:
            le = mapping_set[col]
            input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Predict
    log_rank = model.predict(input_data)
    predicted_rank = np.expm1(log_rank)[0]  # Convert from log

    st.success(f"Predicted Closing Rank: **{int(predicted_rank)}**")

