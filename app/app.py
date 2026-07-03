import os
import streamlit as st
import pandas as pd
import joblib

# --- Path setup (script-relative, works locally and on Streamlit Cloud) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

st.set_page_config(page_title="Customer Churn & Segmentation", layout="wide")
st.title("📊 Customer Segmentation & Churn Prediction Dashboard")

# --- Load models once, cached across reruns (performance fix) ---
@st.cache_resource
def load_models():
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'churn_rf_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'rfm_scaler.pkl'))
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))
    return rf_model, scaler, kmeans, model_columns

rf_model, scaler, kmeans, model_columns = load_models()

uploaded_file = st.file_uploader("Upload customer CSV", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # --- Validate required columns exist before proceeding ---
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    required_cols = service_cols + ['tenure', 'MonthlyCharges', 'TotalCharges']
    missing_cols = [c for c in required_cols if c not in data.columns]

    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
        st.stop()

    # --- Force correct dtypes for known numeric columns (guards against CSV quirks) ---
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
    data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')
    data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

    # --- Drop rows where core numeric inputs are unusable, warn the user ---
    before_rows = len(data)
    data = data.dropna(subset=['tenure', 'MonthlyCharges']).reset_index(drop=True)
    dropped = before_rows - len(data)
    if dropped > 0:
        st.warning(f"Dropped {dropped} row(s) with invalid/missing tenure or MonthlyCharges values.")

    if data.empty:
        st.error("No valid rows remaining after cleaning. Please check your CSV.")
        st.stop()

    # --- Segmentation (RFM + KMeans) ---
    st.subheader("Customer Segments")
    data['ServiceCount'] = (data[service_cols] == 'Yes').sum(axis=1)

    # Column names must match exactly what the scaler was fit on
    rfm_input = data[['tenure', 'ServiceCount', 'MonthlyCharges']].copy()
    rfm_input.columns = ['Recency', 'Frequency', 'Monetary']

    rfm_scaled = scaler.transform(rfm_input)
    data['Cluster'] = kmeans.predict(rfm_scaled)

    st.bar_chart(data['Cluster'].value_counts())
    st.dataframe(data[['customerID', 'tenure', 'ServiceCount', 'MonthlyCharges', 'Cluster']])

    # --- Churn Prediction ---
    st.subheader("Churn Predictions")

    df_pred = data.drop(columns=['customerID']) if 'customerID' in data.columns else data.copy()
    if 'Churn' in df_pred.columns:
        df_pred = df_pred.drop(columns=['Churn'])  # never a feature, only a label if present

    cat_cols = df_pred.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df_pred, columns=cat_cols, drop_first=True)

    # Align columns to match training data exactly (drops extras, fills missing with 0)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    predictions = rf_model.predict(df_encoded)
    probabilities = rf_model.predict_proba(df_encoded)[:, 1]

    data['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    data['Churn_Probability'] = probabilities.round(3)

    st.dataframe(data[['customerID', 'Churn_Prediction', 'Churn_Probability']])

    # --- Retention Recommendations ---
    st.subheader("Retention Recommendations")
    high_risk = data[data['Churn_Probability'] > 0.5]
    st.write(f"⚠️ {len(high_risk)} customers flagged as high churn risk.")
    st.dataframe(high_risk[['customerID', 'Churn_Probability', 'Cluster']])
else:
    st.info("Upload a CSV file to get started.")