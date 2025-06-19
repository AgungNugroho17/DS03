import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# Judul halaman
st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan data pribadi untuk memprediksi tingkat obesitas Anda.")

# Form input
gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
age = st.number_input("Usia", min_value=10, max_value=100)
height = st.number_input("Tinggi Badan (meter)", format="%.2f")
weight = st.number_input("Berat Badan (kg)")
family_history = st.selectbox("Riwayat keluarga dengan kelebihan berat badan?", ["yes", "no"])
favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Frekuensi konsumsi sayur", 1.0, 3.0, step=0.1)
ncp = st.slider("Jumlah makan besar per hari", 1.0, 4.0, step=0.1)
caec = st.selectbox("Kebiasaan ngemil", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Konsumsi air per hari", 1.0, 3.0, step=0.1)
scc = st.selectbox("Pantau asupan kalori?", ["yes", "no"])
faf = st.slider("Frekuensi aktivitas fisik", 0.0, 3.0, step=0.1)
tue = st.slider("Durasi penggunaan teknologi", 0.0, 2.0, step=0.1)
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Mapping ke angka sesuai preprocessing
map_binary = {"yes": 1, "no": 0}
map_gender = {"Female": 0, "Male": 1}
map_caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
map_calc = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
map_mtrans = {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4
}

# Ubah input ke array
input_data = np.array([[
    map_gender[gender],
    age,
    height,
    weight,
    map_binary[family_history],
    map_binary[favc],
    fcvc,
    ncp,
    map_caec[caec],
    map_binary[smoke],
    ch2o,
    map_binary[scc],
    faf,
    tue,
    map_calc[calc],
    map_mtrans[mtrans]
]])

# Prediksi
if st.button("Prediksi"):
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    kelas = [
        "Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II",
        "Obesity Type I", "Obesity Type II", "Obesity Type III"
    ]
    st.success(f"Hasil prediksi: **{kelas[pred]}**")
