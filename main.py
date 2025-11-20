import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.api import ExponentialSmoothing
from PIL import Image

# Page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Aplikasi Forecasting Penjualan Kaos"
)

# Top banner
image_banner = Image.open('ds.png')  # Replace with your image file
st.image(image_banner, width='stretch')

st.title("📈Forecasting Penjualan Kaos")
st.write("Menggunakan metode Triple Exponential Smoothing (Holt-Winters) untuk memprediksi penjualan kaos pada toko Kaosdisablon")

# Pemetaan file csv
FILE_MAP = {
    "Kaos Pendek Dewasa": "pendek_dewasa_baru.csv",
    "Kaos Panjang Dewasa": "panjang_dewasa_baru.csv",
    "Kaos T-Shirt Anak": "tshirt_anak_baru.csv"
}

TREND_TYPE = "mul"
SEASONAL_TYPE = "mul"

# FUNGSI PEMUATAN DATA (Satu Fungsi untuk Semua Kategori)
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_path, category_name):
    """Memuat data dari file path dan mengembalikan series dengan frekuensi bulanan."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"⚠️ File data '{file_path}' untuk {category_name} TIDAK DITEMUKAN.")
        return pd.Series()
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        return pd.Series()

    # Asumsi kolom pertama adalah Waktu, kolom kedua adalah Nilai
    date_col = [col for col in df.columns if 'date' in col.lower()][0]
    value_col = [col for col in df.columns if col not in [date_col]][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    series = pd.to_numeric(df[value_col], errors='coerce').dropna()
    series = series.asfreq('MS')

    st.sidebar.success(f"Data {category_name} berhasil dimuat.")
    return series.rename(category_name)

# Fungsi Modelling dan Forecasting
@st.cache_resource
def run_holt_winters_forecast(series, n_months):
    """Melakukan fitting model Holt-Winters dan membuat forecast."""
    try:
        # Model Holt-Winters: Trend Multiplikatif, Seasonal Multiplikatif, m=12
        model = ExponentialSmoothing(
            series,
            seasonal_periods=12,
            trend=TREND_TYPE,
            seasonal=SEASONAL_TYPE,
            initialization_method="estimated"
        )
        
        # Fitting model
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(n_months)
        
        return model_fit, forecast
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal melatih model. Error: {e}")
        return None, None


# Sidebar
st.sidebar.header("Konfigurasi Forecasting")

# Input Kategori
selected_category = st.sidebar.selectbox(
    "Pilih Kategori Produk",
    list(FILE_MAP.keys())
)
file_path = FILE_MAP[selected_category]

# Input jumlah bulan forecast
forecast_months = st.sidebar.slider(
    "Tentukan Jumlah Bulan Forecast",
    min_value=1,
    max_value=36, 
    value=12,
    step=1
)

# Main Tab
# Pemuatan data
series = load_and_preprocess_data(file_path, selected_category)

if series.empty:
    st.error("Aplikasi tidak dapat melanjutkan karena data historis kosong atau gagal dimuat.")
    st.stop()


# --- TAB MENU ---
tab1, tab2, tab3 = st.tabs(["1. Dataset Historis", "2. Grafik Historis", "3. Hasil Forecast"])

# --- TAB 1: DATASET ---
with tab1:
    st.header(f"Dataset Historis ({selected_category})")
    st.write(f"Data historis telah diubah menjadi format bulanan (Monthly Time Series).")
    st.dataframe(series.to_frame().tail(24).rename(columns={selected_category: "Unit Penjualan"}), width='stretch') 
    
    st.subheader("Informasi Data")
    # Bagian Periode dan Jumlah Data
    st.write(f"Periode data historis: **{series.index.min().strftime('%Y-%m-%d')}** s/d **{series.index.max().strftime('%Y-%m-%d')}**")
    st.write(f"Total data bulanan: **{len(series)}** bulan")

    st.subheader("Analisis Statistik Kunci")
    col1, col2, col3, col4 = st.columns(4)

    # Rata-rata (Warna Biru / Info)
    with col1:
        mean_val = series.mean()
        with st.container(border=True): # Menggunakan container dengan border
            st.metric(label="Rata-Rata Penjualan", value=f"{mean_val:,.0f} Unit")
        
    
    # Maksimum (Warna Hijau / Success)
    with col2:
        max_val = series.max()
        with st.container(border=True):
            st.metric(label="Penjualan Tertinggi", value=f"{max_val:,.0f} Unit", delta="Max")
          
        
    # Minimum (Warna Kuning / Warning)
    with col3:
        min_val = series.min()
        with st.container(border=True):
            st.metric(label="Penjualan Terendah", value=f"{min_val:,.0f} Unit", delta="Min")
            

    # Standar Deviasi (Warna Merah / Error - untuk volatilitas)
    with col4:
        std_val = series.std()
        with st.container(border=True):
            st.metric(label="Standar Deviasi", value=f"{std_val:,.0f} Unit", help="Menunjukkan seberapa bervariasinya data penjualan dari rata-rata.")
        

# --- TAB 2: GRAFIK TREND ---
with tab2:
    st.header("Grafik Penjualan Historis")
    
    fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
    ax_trend.plot(series, label=f'Penjualan Bulanan ({selected_category})', color='blue')
    ax_trend.set_title(f"Penjualan Bulanan - {selected_category}")
    ax_trend.set_xlabel("Waktu")
    ax_trend.set_ylabel("Unit Penjualan")
    ax_trend.legend()
    ax_trend.grid(True)
    st.pyplot(fig_trend)

# --- TAB 3: HASIL FORECAST ---
with tab3:
    st.header(f"Plot Hasil Forecast {forecast_months} Bulan")
    
    # Tombol untuk memicu pelatihan model
    if st.button("Jalankan Forecasting", type="primary"):
        
        # Run model (model dilatih dari awal setiap tombol ditekan)
        model_fit, forecast = run_holt_winters_forecast(series, forecast_months)

        if model_fit is None:
            st.warning("Forecast tidak dapat dijalankan karena model gagal dibuat.")
            st.stop()

        # --- Visualisasi Hasil ---
        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
        
        # Plot data historis
        ax_forecast.plot(series, label='Data Historis (Bulanan)', color='blue')
        
        # Plot garis batas peramalan di titik terakhir data historis
        # Gunakan index[-1] untuk mendapatkan titik waktu terakhir dari data series
        ax_forecast.axvline(x=series.index[-1], color='grey', linestyle=':', label='Batas Peramalan Masa Depan')
        
        # Plot hasil forecast
        ax_forecast.plot(forecast, label=f'Forecast {forecast_months} Bulan', color='green', linewidth=2)
        
        ax_forecast.set_title(f"Hasil Forecasting Holt-Winters - {selected_category}")
        ax_forecast.set_xlabel("Waktu")
        ax_forecast.set_ylabel("Unit Penjualan")
        ax_forecast.legend()
        ax_forecast.grid(True)
        st.pyplot(fig_forecast)
        
        # --- Tampilkan Hasil Forecast dalam Tabel ---
        forecast_df = pd.DataFrame({
            'Bulan': forecast.index,
            'Forecast_Unit_Penjualan': forecast.round(0).values
        })
        forecast_df['Bulan'] = forecast_df['Bulan'].dt.strftime('%Y-%m-%d')
        
        st.subheader("Data Hasil Forecast")
        st.dataframe(forecast_df, hide_index=True, width='stretch')
        
        # --- Tambahkan Download Button ---
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Forecast (CSV)",
            data=csv,
            file_name=f'forecast_results_{selected_category.replace(" ", "_")}.csv',
            mime='text/csv',
        )
        
    else:

        st.info("Pilih kategori di sidebar dan tekan tombol 'Jalankan Forecasting' untuk melihat hasilnya.")

