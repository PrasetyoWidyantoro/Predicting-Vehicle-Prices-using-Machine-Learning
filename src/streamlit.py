import streamlit as st
import requests

# Judul Aplikasi
st.title("Vehicle Selling Price Predictor")

# Judul Pendamping
st.write("Welcome to the Vehicle Selling Price Predictor. Enter the details below to get a price prediction for your vehicle.")

# Input Fields
year = st.slider("Year", min_value=2003, max_value=2018, value=2010)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.32, max_value=92.6, step=0.1)
kms_driven = st.number_input("Kms Driven", min_value=500, max_value=500000, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 3])

# Tombol untuk Prediksi
if st.button("Predict"):
    # Membentuk data untuk dikirim ke API FastAPI
    data = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner
    }

    # Mengirim permintaan POST ke API FastAPI
    response = requests.post("http://api:8080/predict/", json=data)
    - #khusus kalau mau pakai 2 terminal di jupyterlab atau powershell ganti laman diatas dengan laman dibawah ini
    #response = requests.post("http://localhost:8080/predict/", json=data) 
    

    # Menampilkan hasil prediksi dari API
    if response.status_code == 200:
        result = response.json()
        selling_price = result.get("Selling_Price")
        st.success(f"Predicted Selling Price: {selling_price:.2f} lakhs")
    else:
        st.error("Prediction Error. Please check your input data.")
