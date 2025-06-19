import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# Judul
st.title("Prediksi Obesitas - Versi Minimalis")
st.write("Masukkan data berikut untuk memprediksi tingkat obesitas Anda.")

# Input Fitur (6 fitur terbaik)
weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, step=0.1)
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
age = st.number_input("Usia (tahun)", min_value=5, max_value=100, step=1)
faf = st.slider("Frekuensi Aktivitas Fisik (0 = tidak aktif, 3 = sangat aktif)", 0.0, 3.0, step=0.1)
ch2o = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, step=0.1)
fcvc = st.slider("Frekuensi Konsumsi Sayur (1 - 3)", 1.0, 3.0, step=0.1)

# Gabungkan input ke array
input_data = np.array([[age, height, weight, faf, ch2o, fcvc]])

# Prediksi
if st.button("Prediksi"):
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    kelas = [
        "Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II",
        "Obesity Type I", "Obesity Type II", "Obesity Type III"
    ]
    st.success(f"Hasil prediksi: **{kelas[pred]}**")
