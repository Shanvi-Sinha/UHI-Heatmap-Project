
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Urban Heat Island Mapping",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🌍 Urban Heat Island Mapping")
st.subheader("Deep Learning-Based Temperature Prediction Using Landsat 9")

st.write(
    "An interactive visualization of Land Surface Temperature "
    "using multispectral satellite imagery and a CNN model."
)

# Load saved project outputs
predicted = np.load(os.path.join(BASE_DIR, "Predicted_Heatmap.npy"))
actual = np.load(os.path.join(BASE_DIR, "Actual_LST.npy"))
error = np.load(os.path.join(BASE_DIR, "Error_Map.npy"))
metrics = pd.read_csv(os.path.join(BASE_DIR, "Evaluation_Metrics.csv"))

# Performance metrics
st.header("Model Performance")

cols = st.columns(3)

for col, (_, row) in zip(cols, metrics.iterrows()):
    col.metric(row["Metric"], round(row["Value"], 4))

# Heat maps
st.header("Temperature Visualization")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im1 = axes[0].imshow(actual, cmap="hot", vmin=30, vmax=60)
axes[0].set_title("Actual LST")
plt.colorbar(im1, ax=axes[0], label="Temperature (°C)")

im2 = axes[1].imshow(predicted, cmap="hot", vmin=30, vmax=60)
axes[1].set_title("CNN Predicted LST")
plt.colorbar(im2, ax=axes[1], label="Temperature (°C)")

im3 = axes[2].imshow(error, cmap="coolwarm", vmin=-5, vmax=5)
axes[2].set_title("Prediction Error")
plt.colorbar(im3, ax=axes[2], label="Error (°C)")

st.pyplot(fig)
plt.close(fig)

# Temperature statistics
st.header("Temperature Analysis")

col1, col2 = st.columns(2)

col1.metric("Actual Mean Temperature", f"{np.nanmean(actual):.2f} °C")
col2.metric("Predicted Mean Temperature", f"{np.nanmean(predicted):.2f} °C")

st.write("### Predicted Temperature Distribution")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(predicted[np.isfinite(predicted)], bins=40)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Number of Pixels")
ax.set_title("CNN Predicted Temperature Distribution")

st.pyplot(fig)
plt.close(fig)

st.caption("Project: Urban Heat Island Mapping | Landsat 9 | CNN")
