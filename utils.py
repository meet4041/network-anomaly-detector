import pandas as pd
import matplotlib.pyplot as plt

def plot_mse_distribution(mse, threshold):
    fig, ax = plt.subplots()
    ax.hist(mse, bins=50, alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.5f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Frequency")
    ax.set_title("Autoencoder Reconstruction Error")
    ax.legend()
    return fig

def generate_result_df(preds_if, preds_ae, y):
    return pd.DataFrame({
        "IsolationForest": preds_if,
        "Autoencoder": preds_ae,
        "ActualLabel": y
    })
