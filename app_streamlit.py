import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model dan encoder
with open("xgb_model_fixed.pkl", "rb") as f:
    model = pickle.load(f)

with open("oh_encoder_fixed.pkl", "rb") as f:
    oh_encoder = pickle.load(f)

with open("scaler_fixed.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("object_cols_fixed.pkl", "rb") as f:
    object_cols = pickle.load(f)

with open("X_train_columns_fixed.pkl", "rb") as f:
    X_columns = pickle.load(f)

# Fitur numerik
numerical_features = ['lat', 'long', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2',
                      'carports', 'maid_bedrooms', 'maid_bathrooms', 'floors', 'building_age',
                      'year_built', 'garages']

# Fungsi prediksi
def predict_house_price(input_data: dict):
    num_df = pd.DataFrame([input_data], columns=numerical_features)
    cat_df = pd.DataFrame([input_data], columns=object_cols)
    num_scaled = scaler.transform(num_df)
    num_scaled_df = pd.DataFrame(num_scaled, columns=numerical_features)
    cat_encoded = oh_encoder.transform(cat_df)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=oh_encoder.get_feature_names_out())
    final_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    missing_cols = set(X_columns) - set(final_df.columns)
    for col in missing_cols:
        final_df[col] = 0
    final_df = final_df[X_columns]
    prediction = model.predict(final_df)[0]
    return prediction

# Streamlit UI
st.title("Prediksi Harga Rumah di Jabodetabek")
st.markdown("Silakan masukkan informasi properti untuk memprediksi harga:")

with st.form("form"):
    title = st.text_input("Judul Iklan", "Rumah Di Tanggerang Dekat Tol Karawaci 1 Idaman Muslim Dekat Masjid View Danau")
    address = st.text_input("Alamat", "Tangerang Kota, Tangerang")
    district = st.text_input("Kecamatan", "Tangerang Kota")
    city = st.selectbox("Kota", ["Bekasi", "Depok", "Tangerang", "Jakarta", "Bogor"])
    lat = st.number_input("Latitude", value=-6.17)
    long = st.number_input("Longitude", value=1066)
    facilities = st.text_input("Fasilitas", "Masjid,  Taman, Tempat Jemuran, Lapangan Bulu Tangkis, Kitchen Set, Keamanan 24 jam, Wastafel, Track Lari, Taman,  One Gate System")
    property_type = st.selectbox("Tipe Properti", ["rumah", "apartemen"])
    bedrooms = st.number_input("Kamar Tidur", 1, 10, 6)
    bathrooms = st.number_input("Kamar Mandi", 1, 10, 3)
    land_size_m2 = st.number_input("Luas Tanah (m2)", value=180)
    building_size_m2 = st.number_input("Luas Bangunan (m2)", value=140)
    carports = st.number_input("Carport", 0, 5, 1)
    certificate = st.selectbox("Sertifikat", ["shm", "hgb", "lainnya"])
    electricity = st.selectbox("Listrik", ["1300 VA", "2200 VA", "3500 VA", "4400 VA", "5500 VA"])
    maid_bedrooms = st.number_input("Kamar Pembantu", 0, 3, 0)
    maid_bathrooms = st.number_input("Kamar Mandi Pembantu", 0, 3, 0)
    floors = st.number_input("Jumlah Lantai", 1, 5, 2)
    building_age = st.number_input("Usia Bangunan (tahun)", 0, 100, 2)
    year_built = st.number_input("Tahun Dibangun", 1950, 2025, 2023)
    property_condition = st.selectbox("Kondisi Properti", ["baru", "bagus", "butuh renovasi"])
    building_orientation = st.selectbox("Arah Bangunan", ["utara", "selatan", "barat", "timur"])
    garages = st.number_input("Jumlah Garasi", 0, 3, 1)
    furnishing = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])

    submitted = st.form_submit_button("Prediksi Harga")

    if submitted:
        input_data = {
            'title': title,
            'address': address,
            'district': district,
            'city': city,
            'lat': lat,
            'long': long,
            'facilities': facilities,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'land_size_m2': land_size_m2,
            'building_size_m2': building_size_m2,
            'carports': carports,
            'certificate': certificate,
            'electricity': electricity,
            'maid_bedrooms': maid_bedrooms,
            'maid_bathrooms': maid_bathrooms,
            'floors': floors,
            'building_age': building_age,
            'year_built': year_built,
            'property_condition': property_condition,
            'building_orientation': building_orientation,
            'garages': garages,
            'furnishing': furnishing,
            'url': "placeholder"  # dummy value
        }

        predicted_price = predict_house_price(input_data)
        st.success(f"Prediksi Harga Rumah: Rp {predicted_price:,.0f}")
