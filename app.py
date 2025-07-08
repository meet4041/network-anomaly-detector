import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from models import get_isolation_forest_model, get_autoencoder_model, predict_autoencoder
from utils import plot_mse_distribution, generate_result_df
from config import CATEGORICAL_COLUMNS

st.set_page_config(page_title="Network Anomaly Detector", layout="centered")
st.title("Network Traffic Anomaly Detection")

uploaded_file = st.file_uploader("Upload Network Traffic CSV (KDD format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X_scaled, y = preprocess_data(df)

    # Isolation Forest
    st.subheader("Isolation Forest Detection")
    iso_model = get_isolation_forest_model()
    preds_if = iso_model.fit_predict(X_scaled)
    preds_if = [0 if x == 1 else 1 for x in preds_if]
    st.success(f"Isolation Forest detected {sum(preds_if)} anomalies")

    # Autoencoder
    st.subheader("Autoencoder Detection")
    autoencoder = get_autoencoder_model(X_scaled.shape[1])
    preds_ae, mse, threshold = predict_autoencoder(autoencoder, X_scaled)
    st.success(f"Autoencoder detected {sum(preds_ae)} anomalies")

    # MSE Plot
    st.subheader("MSE Distribution (Autoencoder)")
    fig = plot_mse_distribution(mse, threshold)
    st.pyplot(fig)

    # Results
    st.subheader("Predictions")
    results_df = generate_result_df(preds_if, preds_ae, y)
    st.dataframe(results_df.head(20))
    st.download_button("Download Results", results_df.to_csv(index=False), "predictions.csv")

else:
    st.info("Upload a CSV file to begin.")
