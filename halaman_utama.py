import streamlit as st
import pandas as pd
import joblib

from feature_importance import show_feature_importance_page
from overview import show_overview
from song_recommendation import show_song_recommendation
from favorite_song_prediction import show_label_analysis

st.set_page_config(page_title="Spotify Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")  # Gunakan path relatif jika file ada di repo yang sama
    return df

df = load_data()

model = joblib.load("model_spotify.pkl")  # Path relatif
fitur_model = [
    "popularity", "duration_ms", "explicit", "danceability", "energy",
    "key", "loudness", "mode", "speechiness"
]

st.sidebar.title("Spotify Dashboard")
page = st.sidebar.radio("Pilih Halaman", [
    "Overview Dashboard",
    "Aplikasi Prediksi Kesukaan Lagu Spotify",
    "Rekomendasi Lagu",
    "Feature Importance"
])

if page == "Overview Dashboard":
    show_overview(df)
elif page =
