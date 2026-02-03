import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained Gradient Boosting model and the scaler
try:
    gbm_model = joblib.load('/content/gbmreg_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'gbmreg_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

st.title('Insurance Charges Prediction with Gradient Boosting')
st.subheader('Enter the Following details to get a prediction:')

# Get user input for features
claim_amount = st.number_input("Claim Amount", min_value=0.0, format="%.2f", value=20000.00)
past_consultations = st.number_input("Number of Past Consultations", min_value=0, value=10)
hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, format="%.2f", value=5000000.00)
annual_salary = st.number_input("Annual Salary", min_value=0.0, format="%.2f", value=60000000.00)
children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Is the person a smoker?", ['No', 'Yes'])

# Encode smoker input
smoker_encoded = 1 if smoker == 'Yes' else 0

# Create a DataFrame from user inputs
user_input_df = pd.DataFrame({
    'claim_amount': [claim_amount],
    'past_consultations': [past_consultations],
    'hospital_expenditure': [hospital_expenditure],
    'annual_salary': [annual_salary],
    'children': [children],
    'smoker': [smoker_encoded]
})

# Ensure the columns are in the same order as during training
# This list should exactly match the features in your X_train/X data used for scaler.fit_transform and model.fit
# Based on the notebook, the final 'x' dataframe had these columns in this order after dropping 'region' and 'sex'
expected_features_order = ['claim_amount', 'past_consultations', 'hospital_expenditure', 'annual_salary', 'children', 'smoker']

# Reorder user input to match the training order
user_input_df = user_input_df[expected_features_order]

# Scale the numerical features
# The scaler was fitted on the entire 'x' DataFrame, which includes both numerical and encoded categorical features like 'smoker'.
# So, we scale the entire preprocessed DataFrame. If 'smoker' was not scaled during training, this needs adjustment.
# Based on the notebook, `X = scaler.fit_transform(x)` means all columns in `x` (which included 'smoker') were scaled.
scaled_input = scaler.transform(user_input_df)

# Make prediction
if st.button('Predict Charges'):
    try:
        prediction = gbm_model.predict(scaled_input)
        st.success(f'Predicted Insurance Charges: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
