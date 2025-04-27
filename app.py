# import streamlit as st
# import pandas as pd
# import joblib
# import os

# # Set working directory
# os.chdir(r"C:\Users\ACER\OneDrive\Desktop\PGM_PROJECT")
# print(f"Current working directory: {os.getcwd()}")  # Debugging output

# # Load the pre-trained model
# MODEL_PATH = "loan_approval_model.pkl"

# # Verify the model file exists before loading
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
# model = joblib.load(MODEL_PATH)
# print("Model loaded successfully.")  # Debugging output

# # Streamlit app
# st.title("Loan Approval Prediction")
# st.write("Enter the applicant's details to predict loan approval.")

# # Input fields for features
# gender = st.selectbox("Gender", ["Male", "Female"])
# married = st.selectbox("Married", ["Yes", "No"])
# dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
# education = st.selectbox("Education", ["Graduate", "Not Graduate"])
# self_employed = st.selectbox("Self Employed", ["Yes", "No"])
# applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
# coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
# loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
# loan_amount_term = st.selectbox("Loan Amount Term (days)", [360, 180, 480, 300, 240, 120])
# credit_history = st.selectbox("Credit History", [1, 0])
# property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# # Prediction button
# if st.button("Predict"):
#     # Prepare input data
#     input_data = {
#         'Gender': 1 if gender == "Male" else 0,
#         'Married': 1 if married == "Yes" else 0,
#         'Dependents': 3 if dependents == "3+" else int(dependents),
#         'Education': 0 if education == "Graduate" else 1,
#         'Self_Employed': 1 if self_employed == "Yes" else 0,
#         'ApplicantIncome': applicant_income,
#         'CoapplicantIncome': coapplicant_income,
#         'LoanAmount': loan_amount,
#         'Loan_Amount_Term': loan_amount_term,
#         'Credit_History': credit_history,
#         'Property_Area': {"Urban": 0, "Semiurban": 1, "Rural": 2}[property_area]
#     }
    
#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Make prediction
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0]

#     # Display result
#     result = "Approved" if prediction == 1 else "Denied"
#     # confidence = probability[prediction] * 100
#     st.write(f"Prediction: **{result}**")
#     # st.write(f"Confidence: {confidence:.2f}%")


import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Set working directory
os.chdir(r"C:\Users\ACER\OneDrive\Desktop\PGM_PROJECT")
st.write(f"Current working directory: {os.getcwd()}")

# Load the pre-trained model
MODEL_PATH = "loan_approval_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
model = joblib.load(MODEL_PATH)
st.write("Model loaded successfully.")

# Streamlit app
st.title("Loan Approval Prediction")
st.markdown("Enter the applicant's details to predict loan approval.")

# Layout with columns for inputs
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
    loan_amount_term = st.selectbox("Loan Amount Term (days)", [360, 180, 480, 300, 240, 120])
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': {"Urban": 0, "Semiurban": 1, "Rural": 2}[property_area]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Display result
    result = "Approved" if prediction == 1 else "Denied"
    confidence = probability[prediction] * 100
    st.subheader("Prediction Result")
    st.write(f"Prediction: **{result}**")
    st.write(f"Confidence: {confidence:.2f}%")
    # Temporary debugging output
    st.write(f"Debug: Probabilities=[Denied: {probability[0]:.4f}, Approved: {probability[1]:.4f}]")

    # Visualization 1: Decision Tree Diagram (using one tree from Random Forest)
    st.subheader("Decision Tree Structure (Sample Tree)")
    fig, ax = plt.subplots(figsize=(20, 12))  # Increased size
    plot_tree(model.estimators_[0], feature_names=input_df.columns, class_names=['Denied', 'Approved'], 
              filled=True, rounded=True, ax=ax, fontsize=12, max_depth=3)  # Larger font, limited depth
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)  # Maximize display width

    # Visualization 2: Gauge Chart for Prediction Confidence
    st.subheader("Prediction Confidence")
    gauge_value = confidence
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': f"Confidence ({result})"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50" if result == "Approved" else "#FF6B6B"},
            'steps': [
                {'range': [0, 50], 'color': "#FFE6E6"},
                {'range': [50, 100], 'color': "#E6FFE6"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'value': gauge_value
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Visualization 3: Feature Importance Bar Chart
    st.subheader("Feature Importance")
    feature_importances = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    fig_bar = px.bar(feature_importances, x="Feature", y="Importance",
                     title="Feature Importance in Random Forest",
                     color="Importance", color_continuous_scale="Blues")
    fig_bar.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model: Random Forest | © 2025")