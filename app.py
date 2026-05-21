import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load model dan scaler yang sudah disimpan
model = joblib.load('best_house_price_model.pkl')
scaler = joblib.load('house_price_scaler.pkl')

# 2. Judul Aplikasi
st.title('🏠 Prediksi Harga Rumah')
st.write('Masukkan detail properti di bawah ini untuk mendapatkan estimasi harga rumah.')

# 3. Membuat Input Form
with st.form("prediction_form"):
    square_footage = st.number_input('Luas Bangunan (Square Footage)', min_value=100, max_value=10000, value=2000)
    num_bedrooms = st.slider('Jumlah Kamar Tidur', 1, 10, 3)
    num_bathrooms = st.slider('Jumlah Kamar Mandi', 1, 5, 2)
    year_built = st.number_input('Tahun Dibangun', min_value=1800, max_value=2024, value=2000)
    lot_size = st.number_input('Luas Tanah (Lot Size)', min_value=0.1, max_value=10.0, value=1.0)
    garage_size = st.selectbox('Kapasitas Garasi', [0, 1, 2, 3])
    neighborhood_quality = st.slider('Kualitas Lingkungan (1-10)', 1, 10, 5)
    
    submit_button = st.form_submit_button(label='Prediksi Harga')

# 4. Logika Prediksi
if submit_button:
    # Menyusun input menjadi DataFrame
    input_data = pd.DataFrame([[square_footage, num_bedrooms, num_bathrooms, year_built, 
                                lot_size, garage_size, neighborhood_quality]],
                              columns=['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 
                                       'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'])
    
    # Melakukan Scaling
    input_scaled = scaler.transform(input_data)
    
    # Melakukan Prediksi
    prediction = model.predict(input_scaled)
    
    # Menampilkan Hasil
    st.success(f'Estimasi Harga Rumah Anda adalah: **${prediction[0]:,.2f}**')

# Petunjuk menjalankan
# Untuk menjalankan aplikasi ini di Colab, Anda perlu menggunakan 'localtunnel' atau 'ngrok'.
print("File app.py berhasil dibuat. Gunakan localtunnel untuk menjalankannya.")
