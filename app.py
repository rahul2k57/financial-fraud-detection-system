#Importing necessary Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1: Setting Up the Page
# Professional layout and title
st.set_page_config(page_title="Fraud Detection", page_icon="🏦", layout="centered")

# 2: Optimized Resource Loading 
# @st.cache_resource ensures that model only loads once, making the app much faster
@st.cache_resource
def load_assets():
    """Load the trained Random Forest model and the Scaler."""
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# 3: Header and the Information
st.title("🛡️ Financial Fraud Detection System")
st.markdown("""
This system uses a Random Forest Classifier trained on the PaySim dataset to identify fraudulent transactions in real-time.
""")

# 4: Forms for the data entry

# Using st.form prevents the page from refreshing after every single input
with st.form("transaction_data"):
    st.subheader("Transaction Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tx_type = st.selectbox("Method", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'])
        tx_amount = st.number_input("Amount ($)", min_value=0.01, format="%.2f")
        tx_step = st.number_input("Time Step (1-744)", min_value=1, step=1)
    
    with col2:
        sender_old = st.number_input("Sender's Old Balance", min_value=0.0, format="%.2f")
        sender_new = st.number_input("Sender's New Balance", min_value=0.0, format="%.2f")
        recipient_old = st.number_input("Recipient's Old Balance", min_value=0.0, format="%.2f")
        recipient_new = st.number_input("Recipient's New Balance", min_value=0.0, format="%.2f")

    submit_button = st.form_submit_button("Analyze Risk Profile")

# 5: Logic and Prediction
if submit_button:
    if model is None or scaler is None:
        st.error("Missing model artifacts! Please ensure .pkl files are in the directory.")
    else:
        # A. Feature Engineering 
        # Creating the logical features which we have created in the main code
        error_orig = sender_new + tx_amount - sender_old
        error_dest = recipient_old + tx_amount - recipient_new
        zero_dest = 1 if (recipient_new == 0 and recipient_old == 0) else 0

        # B. One-Hot Encoding (Converting Categories to 1s and 0s)
        # Must match the exactly the columns: ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        # Note: 'CASH_IN' is the dropped category (drop_first=True)
        # Here 'CASH_IN' is dropped because to run the model faster than usual
        encoding = {'CASH_OUT': 0, 'DEBIT': 0, 'PAYMENT': 0, 'TRANSFER': 0}
        if tx_type in encoding:
            encoding[tx_type] = 1

        # C. Features Assembly
        # The list follows the exact 13-feature order the model is expecting
        final_input = [
            tx_step, tx_amount, sender_old, sender_new, recipient_old, recipient_new,
            error_orig, error_dest, zero_dest,
            encoding['CASH_OUT'], encoding['DEBIT'], 
            encoding['PAYMENT'], encoding['TRANSFER']
        ]

        # D: Predicition
        try:
            # Scaling the features to the standard distribution (0-1 range typically)
            scaled_features = scaler.transform([final_input])
            
            # Getting the raw prediction (0 or 1) and the probability score
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0][1]

            # 6: Ouptut
            st.divider()
            if prediction == 1:
                st.error(f"FRAUD IS DETECTED! (Confidence: {probability*100:.1f}%)")
                st.warning("Anomaly Note: High balance mismatch detected in the origin account.")
            else:
                st.success(f"TRANSACTION IS SAFE (Risk Score: {probability*100:.2f}%)")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")