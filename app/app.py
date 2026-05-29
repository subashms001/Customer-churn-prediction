import streamlit as st
import pandas as pd
import numpy as np
import joblib

model         = joblib.load('model.pkl')
scaler        = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Customer Churn Predictor")
st.markdown("Enter customer details to predict the likelihood of churning.")
st.divider()

# form

col1, col2 = st.columns(2)

with col1:
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    contract         = st.selectbox("Contract Type", 
                                    ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", 
                                    ["DSL", "Fiber optic", "No"])

with col2:
    online_security  = st.selectbox("Online Security",  ["Yes", "No"])
    tech_support     = st.selectbox("Tech Support",     ["Yes", "No"])
    payment_method   = st.selectbox("Payment Method",   
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", 
                                     "Credit card (automatic)"])
    senior_citizen   = st.radio("Senior Citizen", ["No", "Yes"])

def build_input():
    data = {f: 0 for f in feature_names}

    # Numeric fields
    data['tenure']         = tenure
    data['MonthlyCharges'] = monthly_charges
    data['TotalCharges']   = monthly_charges * tenure  # estimate

    # Binary fields
    data['SeniorCitizen']    = 1 if senior_citizen == "Yes" else 0
    data['OnlineSecurity']   = 1 if online_security == "Yes" else 0
    data['TechSupport']      = 1 if tech_support == "Yes" else 0

    # One-hot: Contract
    if contract == "One year":
        data['Contract_One year'] = 1
    elif contract == "Two year":
        data['Contract_Two year'] = 1

    # One-hot: Internet Service
    if internet_service == "Fiber optic":
        data['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        data['InternetService_No'] = 1

    # One-hot: Payment Method
    pm_map = {
        "Mailed check":                   "PaymentMethod_Mailed check",
        "Credit card (automatic)":        "PaymentMethod_Credit card (automatic)",
        "Bank transfer (automatic)":      "PaymentMethod_Bank transfer (automatic)",
    }
    if payment_method in pm_map:
        data[pm_map[payment_method]] = 1

    df = pd.DataFrame([data])

    # Scale numeric columns
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    return df


st.divider()
if st.button("🔮 Predict Churn", use_container_width=True):
    input_df   = build_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK — This customer is likely to churn**")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.markdown("""
        **Suggested Actions:**
        - Offer a discounted annual contract upgrade
        - Proactively reach out with retention offer
        - Offer free tech support trial
        """)
    else:
        st.success(f"✅ **LOW RISK — This customer is likely to stay**")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.markdown("No immediate retention action needed.")

    st.progress(float(probability))
    st.caption(f"Model confidence: {probability:.1%} probability of churning")





