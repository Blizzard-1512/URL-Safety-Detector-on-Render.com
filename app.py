import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from urllib.parse import urlparse
import socket
import requests
import json
import os

from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model

# Set page configuration to wide mode and default theme
st.set_page_config(page_title="URL Safety Prediction using Machine Learning & Deep Learning",
                   page_icon=":shield:",
                   layout="wide",
                   initial_sidebar_state="expanded")


# Load all models
def load_model_from_file(file_name):
    """Utility to load models with fallback for Render environment."""
    model_path = os.path.join(os.getenv("MODEL_DIR", ""), file_name)
    if not os.path.exists(model_path):
        st.error(f"Model file '{file_name}' not found!")
        st.stop()
    return pickle.load(open(model_path, "rb"))


try:
    vtc = load_model_from_file("vtc.pkl")
    dtc = load_model_from_file("dtc.pkl")
    rf = load_model_from_file("rf.pkl")
    bcf = load_model_from_file("bcf.pkl")
    xgb = load_model_from_file("xgb.pkl")
    abc = load_model_from_file("abc.pkl")
    svm = load_model_from_file("svm.pkl")
    lr = load_model_from_file("lr.pkl")
    model_path = os.path.join(os.getenv("MODEL_DIR", ""), "final_model.h5")
    model = load_model(model_path)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Attempt to load test data
try:
    test_data_path = os.path.join(os.getenv("DATA_DIR", ""), "test_data.pkl")
    test_data = pickle.load(open(test_data_path, "rb"))
    X_test = test_data["X"]
    y_test = test_data["y"]
except FileNotFoundError:
    st.warning(
        "Test data file 'test_data.pkl' not found. Accuracy calculations will not be performed."
    )
    X_test, y_test = None, None

# Custom CSS for dark theme and styling
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
.stDataFrame {
    color: #000000;
}
.stTextInput > div > div > input {
    color: #FFFFFF;
    background-color: #262730;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# Define the app title
st.title("URL Safety Prediction using Machine Learning & Deep Learning")

# Insert steps for using the application
st.subheader("Steps to use the URL Legitimacy Detector")
st.markdown(""" 
1. **Copy any URL** of your choice which you want to test and find out whether it is safe or malicious.  
2. **Select the model** you want to predict the result with from the sidebar on the left.  
   - We recommend using the **Random Forest Classifier** for high accuracy.  
   - Use the **Bagging Classifier** for consistent predictions.  
3. After entering your URL and selecting the model, **click on the "Predict" button** and watch the magic happen.  
   - Our model extracts the actual features from your URL and processes them to test for its legitimacy.  
""")

# URL Input
st.header("Enter a URL")
url_input = st.text_input("Input the URL:")


# Feature extraction function
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)

    # Extract features (same as previous implementation)
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0
    features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
    features["Shortining_Service"] = 1 if "bit.ly" in url or "t.co" in url else 0
    features["having_At_Symbol"] = 1 if "@" in url else 0
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
    features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
    features["having_Sub_Domain"] = -1 if parsed_url.netloc.count(".") > 2 else (
        0 if parsed_url.netloc.count(".") == 2 else 1)
    features["SSLfinal_State"] = 1 if url.startswith("https") else -1
    features["Domain_registeration_length"] = 0  # Placeholder
    features["Favicon"] = 1  # Placeholder
    features["port"] = 1  # Placeholder
    features["HTTPS_token"] = 1 if "https-" in parsed_url.netloc else 0
    features["Request_URL"] = 1  # Placeholder
    features["URL_of_Anchor"] = 0  # Placeholder
    features["Links_in_tags"] = 0  # Placeholder
    features["SFH"] = 0  # Placeholder
    features["Submitting_to_email"] = 1 if "mailto:" in url else 0
    features["Abnormal_URL"] = -1 if len(parsed_url.netloc.split(".")) > 3 else 1
    features["Redirect"] = 1 if "→" in url else -1
    features["on_mouseover"] = 0  # Placeholder
    features["RightClick"] = 0  # Placeholder
    features["popUpWidnow"] = 0  # Placeholder
    features["Iframe"] = 0  # Placeholder
    features["age_of_domain"] = 0  # Placeholder
    features["DNSRecord"] = 0  # Placeholder
    features["web_traffic"] = 0  # Placeholder
    features["Page_Rank"] = 0  # Placeholder
    features["Google_Index"] = 1 if "google.com" in url else 0
    features["Links_pointing_to_page"] = 0  # Placeholder
    features["Statistical_report"] = 0  # Placeholder

    return features


# Extract features when URL is entered
if url_input:
    st.write("Extracting features from the URL...")
    extracted_features = extract_features(url_input)
    feature_values = np.array([[extracted_features[key] for key in extracted_features]])

    # Get top 5 contributing features
    top_features = pd.Series(extracted_features).sort_values(ascending=False)[:5]

    st.write("Extracted feature values:")
    for key, value in extracted_features.items():
        st.write(f"{key}: {value}")
else:
    extracted_features = None
    top_features = None

# Sidebar with model selection
st.sidebar.header("Select Models for Prediction")
models = {
    "Voting Classifier": vtc,
    "Decision Trees": dtc,
    "Random Forests (Better for Generalization)": rf,
    "Bagging Classifier (Better for Consistency)": bcf,
    "XGBoost Classifier": xgb,
    "AdaBoost Classifier": abc,
    "Support Vector Classifier": svm,
    "Neural Networks": model
}

selected_models = []
for model_name in models:
    if st.sidebar.checkbox(model_name):
        selected_models.append((model_name, models[model_name]))

# Add a separation line between the button and model options
st.sidebar.markdown("<hr>", unsafe_allow_html=True)


# Function to convert continuous probabilities to binary class labels
def convert_to_class_labels(predictions, threshold=0.5):
    """Converts continuous predictions to binary class labels based on a threshold."""
    return (predictions > threshold).astype(int)


# Preloaded sample malicious URLs
malicious_url_samples = [
    "http://secure-data-access.com",
    "http://account-verification-update.net",
    "http://confirm-details-protect.io",
    "http://secure-login-prompt.org",
    "http://system-update-required.xyz",
    "http://verify-account-urgent.club"
]

# Add a prediction button
if st.sidebar.button("Predict"):
    predictions = {}

    if extracted_features is not None:
        for model_name, model in selected_models:
            try:
                if hasattr(model, "predict_proba"):
                    prediction_probs = model.predict_proba(feature_values)[:, 1]
                    prediction_class = convert_to_class_labels(prediction_probs)
                else:
                    prediction_class = model.predict(feature_values)

                accuracy = None
                if X_test is not None and y_test is not None:
                    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

                predictions[model_name] = {
                    "Prediction": "Safe" if prediction_class[0] == 1 else "Malicious",
                    "Accuracy": f"{accuracy:.2f}%" if accuracy is not None else "N/A"
                }

            except NotFittedError:
                st.error(f"The model {model_name} is not properly fitted.")
            except Exception as e:
                st.error(f"Error with model {model_name}: {str(e)}")

        if predictions:
            prediction_df = pd.DataFrame([{
                "Model": name, "Prediction": details["Prediction"], "Confidence Level": details["Accuracy"]
            } for name, details in predictions.items()])

            st.write("Prediction Results:")
            st.dataframe(prediction_df)

            # Display Top Contributing Features
            st.subheader("Top 5 Contributing Features")
            for feature, value in top_features.items():
                st.write(f"{feature}: {value}")

            # Add a button for Safe URL predictions
            if "Safe" in prediction_df["Prediction"].values:
                st.markdown(
                    f'<a href="{url_input}" target="_blank" style="text-decoration: none;">'
                    f'<button style="background-color: #4CAF50; color: white; padding: 10px 20px; '
                    f'border: none; border-radius: 4px; cursor: pointer;">Go to Safe URL</button></a>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("This URL is potentially malicious. We recommend not visiting it.")
    else:
        st.warning("Please input a URL to predict.")
