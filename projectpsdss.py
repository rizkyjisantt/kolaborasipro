import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from datetime import datetime

st.set_page_config(page_title="Prediksi Saham")

st.write("""# Aplikasi Prediksi Saham PT. Pertamina Geotermal Energy Tbk (PGEO.JK)""")
st.write("Oleh: Mochammad Rizki Aji Santoso (200411100086)")
st.write("Oleh: Muhammad Unis Iswahyudi (200411100185)")
st.write("----------------------------------------------------------------------------------")



dataset, preprocessing, modeling, implementasi = st.tabs(["Data", "Preprocessing", "Modelling", "Implementation"])

with dataset:
    st.header("Data")
    data = pd.read_csv('https://raw.githubusercontent.com/Mohammad-Unis-Iswahyudi/kolaborasipro/master/PGEO.JK.csv')
    st.write(data)
    st.subheader("Penjelasan :")
    st.write("Data berasal dari finance.yahoo.com ")
    st.write("Data tentang finance dari perusahaan PT. Pertamina Geotermal Energy Tbk")
    st.write("Type datanya dari teks time series")

with preprocessing:
    st.header("Preprocessing Data")
    #st.write("""Preprocessing adalah teknik penambangan data yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien.""")
    
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    target = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    st.write("Min Max Scaler")
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    st.write(features_scaled)

    st.write("Reduksi Dimensi PCA")
    pca = PCA(n_components=6)
    features_pca = pca.fit_transform(features_scaled)
    st.write(features_pca)



    
with modeling:
    st.header("Modelling")
    
    knn_cekbox = st.checkbox("KNN")
    rf_cekbox = st.checkbox("Random Forest")
    decission3_cekbox = st.checkbox("Decission Tree")

    #=========================== Spliting data ======================================


    #============================ Model =================================

    #===================== KNN =======================

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Membuat model KNN
    knn_model = KNeighborsRegressor(n_neighbors=5)

    # Melatih model KNN
    knn_model.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = knn_model.predict(X_test)

    # Menghitung MAPE
    mape_knn = mean_absolute_percentage_error(y_test, y_pred)

    #===================== Random Forest =============
    
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Membuat model Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Melatih model Random Forest Regression
    rf_model.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = rf_model.predict(X_test)

    # Menghitung MAPE
    mape_rf = mean_absolute_percentage_error(y_test, y_pred)


    #===================== Decission tree =============
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

    # Membuat model Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state=42)

    # Melatih model Decision Tree Regression
    dt_model.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = dt_model.predict(X_test)

    # Menghitung MAPE
    mape_decision = mean_absolute_percentage_error(y_test, y_pred)

    st.markdown("---")

    #===================== Cek Box ====================

    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Prediksi menggunakan KNN:")
        # st.warning(knn_accuracy)
        st.warning(f"MAPE  =  {mape_knn}")
        st.markdown("---")
    
    if rf_cekbox:
        st.write("##### Random Forest")
        st.warning("Prediksi menggunakan Random Forest:")
        # st.warning(knn_accuracy)
        st.warning(f"MAPE  =  {mape_rf}")
        st.markdown("---")


    if decission3_cekbox:
        st.write("##### Decission Tree")
        st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
        st.success(f"MAPE = {mape_decision}")

# Implementasi menu
with implementasi:
    st.title('Implementasi')

    # Menambahkan konten prediksi
    st.header("Prediksi Saham")
    st.write("""
        Masukkan tanggal untuk melakukan prediksi harga saham.
    """)

    # Membaca input tanggal
    date_input = st.date_input("Tanggal", datetime.now())

    # Mengubah input tanggal menjadi fitur
    date_features = scaler.transform(array([[0, 0, 0, date_input.day, date_input.month, date_input.year]]))

    # Melakukan reduksi dimensi pada fitur tanggal
    date_features_pca = pca.transform(date_features)

    # Melakukan prediksi menggunakan model KNN
    knn_prediction = knn_model.predict(date_features_pca)
        
    # Melakukan prediksi menggunakan model Random Forest
    rf_prediction = rf_model.predict(date_features_pca)

    # Melakukan prediksi menggunakan model Decision Tree
    dt_prediction = dt_model.predict(date_features_pca)

    st.subheader("Hasil Prediksi Saham")
    st.write("Prediksi menggunakan model KNN:")
    st.write(f"Harga saham: {knn_prediction}")

    st.write("Prediksi menggunakan model Random Forest:")
    st.write(f"Harga saham: {rf_prediction}")

    st.write("Prediksi menggunakan model Decision Tree:")
    st.write(f"Harga saham: {dt_prediction}")

