import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn & Segmentation", layout="wide")
st.title("📊 Customer Segmentation & Churn Prediction Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load saved models
rf_model = joblib.load(os.path.join(MODELS_DIR, 'churn_rf_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'rfm_scaler.pkl'))
kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))

uploaded_file = st.file_uploader("Upload customer CSV", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # --- Segmentation ---
    st.subheader("Customer Segments")
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['ServiceCount'] = (data[service_cols] == 'Yes').sum(axis=1)

    rfm_input = data[['tenure', 'ServiceCount', 'MonthlyCharges']]
    rfm_scaled = scaler.transform(rfm_input)
    data['Cluster'] = kmeans.predict(rfm_scaled)

    st.bar_chart(data['Cluster'].value_counts())
    st.dataframe(data[['customerID', 'tenure', 'ServiceCount', 'MonthlyCharges', 'Cluster']])

    # --- Churn Prediction ---
    st.subheader("Churn Predictions")

    df_pred = data.drop(columns=['customerID']) if 'customerID' in data.columns else data.copy()
    if 'TotalCharges' in df_pred.columns:
        df_pred['TotalCharges'] = pd.to_numeric(df_pred['TotalCharges'], errors='coerce').fillna(0)

    cat_cols = df_pred.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df_pred, columns=cat_cols, drop_first=True)

    # Align columns to match training data exactly
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    predictions = rf_model.predict(df_encoded)
    probabilities = rf_model.predict_proba(df_encoded)[:, 1]

    data['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    data['Churn_Probability'] = probabilities.round(3)

    st.dataframe(data[['customerID', 'Churn_Prediction', 'Churn_Probability']])

    st.subheader("Retention Recommendations")
    high_risk = data[data['Churn_Probability'] > 0.5]
    st.write(f"⚠️ {len(high_risk)} customers flagged as high churn risk.")
    st.dataframe(high_risk[['customerID', 'Churn_Probability', 'Cluster']])
else:
    st.info("Upload a CSV file to get started.")