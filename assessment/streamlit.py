# streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import io

# FastAPI endpoint URLs
FASTAPI_URL_PREDICT = "http://localhost:8000/predict/"
FASTAPI_URL_UPLOAD = "http://localhost:8000/upload/"

st.title("MLP Model Prediction and Visualization")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("Data Preview:")
    st.write(pd.read_csv(uploaded_file).head())

    # Predict button
    st.sidebar.header("Input Features")
    feature1 = st.sidebar.number_input("Feature 1", value=0.0)
    feature2 = st.sidebar.number_input("Feature 2", value=0.0)
    features = {"feature1": feature1, "feature2": feature2}

    if st.sidebar.button("Predict"):
        try:
            response = requests.post(FASTAPI_URL_PREDICT, json={"features": features})
            response.raise_for_status()
            prediction = response.json()["prediction"]
            st.write(f"Prediction: {prediction}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during prediction: {e}")

    # Upload file to get visualizations
    try:
        response = requests.post(FASTAPI_URL_UPLOAD, files={"file": uploaded_file})
        response.raise_for_status()
        plots = response.json()

        st.header("Visualizations")

        # Display visualizations
        def display_image_from_url(url, caption):
            st.image(url, caption=caption)

        display_image_from_url(plots['churn_distribution'], "Churn Distribution")
        display_image_from_url(plots['feature_distributions'], "Feature Distributions")
        display_image_from_url(plots['correlation_matrix'], "Correlation Matrix")
        display_image_from_url(plots['pairplot'], "Pairplot of Features")
        display_image_from_url(plots['boxplots'], "Boxplots of Features")
        display_image_from_url(plots['scatter_plots'], "Scatter Plots of Features vs Churn")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching visualizations: {e}")
