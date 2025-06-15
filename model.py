import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("ataset_final.csv")

# Daftar fitur yang digunakan
fitur_model = [
    "explicit", "danceability", "energy", "mode", "speechiness", 
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]

# Klasifikasi berdasarkan popularitas
def classify_popularity(score):
    return 'Suka' if score >= 40 else 'Tidak Suka'

df['popularity_class'] = df['popularity'].apply(classify_popularity)

# Cek distribusi kelas
popularity_counts = df['popularity_class'].value_counts().reindex(['Suka', 'Tidak Suka'])
print("\nDistribusi Kelas:\n", popularity_counts)

# Pisahkan fitur dan label
X = df[fitur_model]
y = df['popularity_class']

# Label encoding pada target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Tampilkan mapping label
print("\nMapping Label:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Pelatihan model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Pastikan folder penyimpanan ada
folder_path = "model_spotify.pkl"
os.makedirs(folder_path, exist_ok=True)

# Simpan semua objek dengan kompresi
try:
    joblib.dump(model, os.path.join(folder_path, "model_spotify.pkl"), compress=3)
    joblib.dump(scaler, os.path.join(folder_path, "scaler_spotify.pkl"), compress=3)
    joblib.dump(fitur_model, os.path.join(folder_path, "fitur_spotify.pkl"), compress=3)
    joblib.dump(le, os.path.join(folder_path, "label_encoder_spotify.pkl"), compress=3)
    print("\n✅ Model dan komponen berhasil disimpan dengan kompresi di folder FILE_PKL.")
except Exception as e:
    print(f"Error saat menyimpan file: {e}")
