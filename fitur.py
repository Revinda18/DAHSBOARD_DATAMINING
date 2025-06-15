import pandas as pd
import joblib

fitur_model = [
    "popularity", "duration_ms", "explicit", "danceability", "energy",
    "key", "loudness", "mode", "speechiness"
]

# Ubah list menjadi DataFrame dengan satu kolom bernama 'fitur'
df_fitur = pd.DataFrame(fitur_model, columns=['fitur'])

# Simpan DataFrame sebagai file pkl menggunakan joblib
joblib.dump(df_fitur, 'fitur_spotify.pkl')

print("Konversi CSV ke PKL selesai!")
