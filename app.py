import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="UHI Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #111318;
    color: #F5F5F5;
}
[data-testid="stSidebar"] {
    background-color: #191D25;
}
h1, h2, h3 {
    color: #FF9D45;
}
[data-testid="stMetric"] {
    background-color: #202630;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #343B48;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD PROJECT FILES ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    predicted = np.load(os.path.join(BASE_DIR, "Predicted_Heatmap.npy"))
    actual = np.load(os.path.join(BASE_DIR, "Actual_LST.npy"))
    error = np.load(os.path.join(BASE_DIR, "Error_Map.npy"))
    metrics = pd.read_csv(os.path.join(BASE_DIR, "Evaluation_Metrics.csv"))
    return predicted, actual, error, metrics

predicted, actual, error, metrics = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌍 UHI INTELLIGENCE")
st.sidebar.caption("Landsat 9 | Deep Learning")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "Heat Map", "Model Analysis", "Data Insights", "Downloads"]
)

st.sidebar.divider()
st.sidebar.caption("CNN-based Urban Heat Island Mapping")

# ---------------- HOME ----------------

if page == "Home":

    st.title("🌍 Urban Heat Island Intelligence")
    st.subheader("Deep Learning-Based Temperature Prediction Using Landsat 9")

    st.markdown("""
    This application uses multispectral satellite imagery and a
    Convolutional Neural Network (CNN) to estimate Land Surface
    Temperature and visualize spatial temperature patterns.
    """)

    st.divider()

    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Satellite", "Landsat 9")
    c2.metric("Deep Learning Model", "CNN")
    c3.metric("Region Size", "256 × 256 pixels")

    st.header("Our Workflow")

    st.markdown("""
    1. Collect multispectral satellite imagery.
    2. Preprocess and clean Landsat data.
    3. Calculate Land Surface Temperature.
    4. Prepare training patches from six spectral bands.
    5. Train a CNN to predict temperature.
    6. Evaluate the model and visualize predicted temperatures.
    """)

    st.info("Use the navigation panel to explore our heat maps, model performance and data insights.")

# ---------------- HEAT MAP ----------------

elif page == "Heat Map":

    st.title("🔥 Temperature Visualization")
    st.write("Explore actual and CNN-predicted Land Surface Temperature.")

    map_choice = st.selectbox(
        "Select visualization",
        ["CNN Predicted LST", "Actual LST", "Prediction Error"]
    )

    if map_choice == "CNN Predicted LST":
        data = predicted
        cmap = "hot"
        title = "CNN Predicted Temperature"

    elif map_choice == "Actual LST":
        data = actual
        cmap = "hot"
        title = "Actual Land Surface Temperature"

    else:
        data = error
        cmap = "coolwarm"
        title = "Prediction Error"

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#191D25")
    ax.set_facecolor("#191D25")

    if map_choice == "Prediction Error":
        image = ax.imshow(data, cmap=cmap, vmin=-5, vmax=5)
        unit = "Error (°C)"
    else:
        image = ax.imshow(data, cmap=cmap, vmin=30, vmax=60)
        unit = "Temperature (°C)"

    ax.set_title(title, color="white", fontsize=15)
    ax.set_xlabel("Pixel Column", color="white")
    ax.set_ylabel("Pixel Row", color="white")
    ax.tick_params(colors="white")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(unit)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Map represents the selected 256 × 256 pixel region.")

# ---------------- MODEL ANALYSIS ----------------

elif page == "Model Analysis":

    st.title("🧠 CNN Model Analysis")
    st.write("Performance metrics and prediction comparisons.")

    metric_values = dict(zip(metrics["Metric"], metrics["Value"]))

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f'{metric_values["MAE"]:.4f} °C')
    c2.metric("RMSE", f'{metric_values["RMSE"]:.4f} °C')
    c3.metric("R² Score", f'{metric_values["R2 Score"]:.4f}')

    st.divider()

    st.subheader("Actual vs Predicted Temperatures")

    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual_values = actual[mask]
    predicted_values = predicted[mask]

    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(actual_values),
        size=min(5000, len(actual_values)),
        replace=False
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        actual_values[indices],
        predicted_values[indices],
        alpha=0.3,
        s=8
    )

    limits = [
        min(actual_values.min(), predicted_values.min()),
        max(actual_values.max(), predicted_values.max())
    ]

    ax.plot(limits, limits, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("Actual vs Predicted")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Understanding the Metrics")

    st.markdown("""
    - **MAE:** Average absolute difference between actual and predicted temperatures.
    - **RMSE:** Measures prediction error while penalizing larger errors.
    - **R²:** Indicates how much temperature variation is explained by the model.
    """)

# ---------------- DATA INSIGHTS ----------------

elif page == "Data Insights":

    st.title("📊 Temperature Data Insights")

    valid = predicted[np.isfinite(predicted)]

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Temperature", f"{np.mean(valid):.2f} °C")
    col2.metric("Maximum Temperature", f"{np.max(valid):.2f} °C")
    col3.metric("Minimum Temperature", f"{np.min(valid):.2f} °C")

    st.divider()

    st.subheader("Temperature Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid, bins=40)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Number of Pixels")
    ax.set_title("CNN Predicted Temperature Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Explore Hotspots")

    threshold = st.slider(
        "Select hotspot temperature threshold (°C)",
        min_value=30,
        max_value=55,
        value=40
    )

    hotspot_mask = np.isfinite(predicted) & (predicted >= threshold)
    hotspot_count = int(hotspot_mask.sum())
    total_pixels = int(np.isfinite(predicted).sum())
    hotspot_percentage = hotspot_count / total_pixels * 100

    c1, c2 = st.columns(2)
    c1.metric("Hot Pixels", f"{hotspot_count:,}")
    c2.metric("Hot Pixels Percentage", f"{hotspot_percentage:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 6))
    hotspot_map = np.where(hotspot_mask, predicted, np.nan)
    image = ax.imshow(hotspot_map, cmap="hot", vmin=30, vmax=60)
    ax.set_title(f"Areas at or Above {threshold} °C")
    fig.colorbar(image, ax=ax, label="Temperature (°C)")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Hotspot threshold is an exploratory visualization setting, not a formal urban heat island classification.")

# ---------------- DOWNLOADS ----------------

elif page == "Downloads":

    st.title("📁 Project Downloads")
    st.write("Download the generated project outputs.")

    files = {
        "Predicted Heat Map": "Predicted_Heatmap.npy",
        "Actual LST Map": "Actual_LST.npy",
        "Prediction Error Map": "Error_Map.npy",
        "Evaluation Metrics": "Evaluation_Metrics.csv",
        "Heat Map Image": "Final_UHI_Heatmap.png",
        "Trained CNN Model": "UHI_CNN_Model.keras"
    }

    for label, filename in files.items():
        file_path = os.path.join(BASE_DIR, filename)

        with open(file_path, "rb") as f:
            st.download_button(
                label=f"Download {label}",
                data=f.read(),
                file_name=filename,
                key=filename
            )

    st.success("All project outputs are available for download.")

# ---------------- FOOTER ----------------

st.divider()
st.caption("Urban Heat Island Mapping | Landsat 9 | CNN Deep Learning Project")
