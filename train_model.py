import streamlit as st
import pandas as pd
import joblib
from Overview import show_eda
from model_classification import show_model_classification
from CONVERT_TO_PKL.feature_importance import show_feature_importance_page

@st.cache_data
def load_data():
    df = pd.read_pickle(r"dataset_final.pkl")
    return df

df = load_data()

# Load model, scaler, dan fitur yang sudah disimpan
model = joblib.load(r"model_spotify.pkl)
scaler = joblib.load(r"scaler_spotify.pkl")
feature_names = joblib.load(r"fitur_spotify.pkl")

st.sidebar.title("Spotify Dashboard")
page = st.sidebar.radio("Pilih Halaman", ["EDA", "Model Klasifikasi", "Rekomendasi Lagu", "Feature Importance"])

if page == "EDA":
    show_eda(df)
elif page == "Model Klasifikasi":
    show_model_classification(df)
elif page == "Feature Importance":
    show_feature_importance_page(df, model, feature_names)
else:
    st.title(f"Halaman {page} sedang dalam pengembangan")
