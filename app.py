import numpy as np
import pandas as pd
import geopandas as gpd

import folium
import streamlit as st
from streamlit_folium import st_folium

import gdown
import joblib
from sklearn.pipeline import Pipeline

# Configure page layout
st.set_page_config(layout="wide")

# [STREAMLIT] HIDE MENU
hide_menu = """
    <style>
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    </style>
    """
st.markdown(hide_menu, unsafe_allow_html=True)

# [STREAMLIT] ADJUST PADDING
padding = """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    </style>
    """
st.markdown(padding, unsafe_allow_html=True)

# Load functions (cached for performance)
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=15pPzWmcpf-RFyxkoDnZo7DWP6OuaMjKA"
    output = "geodata.parquet"
    gdown.download(url, output, quiet=True)
    geo = gpd.read_parquet("geodata.parquet")
    cdo = geo[geo['adm3_psgc'] == 1030500000][['adm4_en', 'geometry']].rename(columns={'adm4_en': 'barangay'})
    features = pd.read_csv("latest_data.csv")
    return cdo.to_crs(epsg=4326), features

@st.cache_resource 
def load_model():
    return joblib.load("pipeline.joblib")

# Load all data
cdo_gdf, features_df = load_data()
pipeline = load_model()

# Merge features with geometries
sim_data = cdo_gdf.merge(features_df, on='barangay')

# Simulation controls
with st.sidebar:
    st.title("Simulation Controls")
    
    with st.expander("🌱 Vegetation Indices"):
        ndvi_adj = st.slider("NDVI Multiplier", 0.5, 1.5, 1.0, 0.01)
        ndwi_adj = st.slider("NDWI Multiplier", 0.5, 1.5, 1.0, 0.01)
        
    with st.expander("🏗️ Urban Features"):
        ndbi_adj = st.slider("NDBI Multiplier", 0.5, 1.5, 1.0, 0.01)
        built_up_adj = st.slider("Built-up Area Multiplier", 0.5, 1.5, 1.0, 0.01)
        
    with st.expander("🌡️ Temperature"):
        skin_temp_adj = st.slider("Skin Temp (°C)", -5.0, 5.0, 0.0, 0.1)
        temp_2m_adj = st.slider("2m Air Temp (°C)", -5.0, 5.0, 0.0, 0.1)
        
    with st.expander("💧 Humidity"):
        humidity_adj = st.slider("Humidity (%)", -10.0, 10.0, 0.0, 0.1)
        temp_dew_adj = st.slider("Dew Point (°C)", -3.0, 3.0, 0.0, 0.1)
        
    with st.expander("🌤️ Radiation"):
        albedo_adj = st.slider("Albedo", 0.8, 1.2, 1.0, 0.01)
        incoming_sw_adj = st.slider("Solar Radiation (W/m²)", -50.0, 50.0, 0.0, 1.0)
        
    scenario = st.selectbox("Climate Scenario", ["Current", "RCP 4.5", "RCP 8.5"])

# Prediction function
def predict_UHI(data):
    # Create adjusted feature matrix
    X_adj = pd.DataFrame({
        'NDBI': data['NDBI'] * ndbi_adj,
        'NDVI': data['NDVI'] * ndvi_adj,
        'NDWI': data['NDWI'] * ndwi_adj,
        'NO2': data['NO2'],  # No adjustment
        'albedo': data['albedo'] * albedo_adj,
        'aspect': data['aspect'],
        'built_up': data['built_up'] * built_up_adj,
        'elevation': data['elevation'],
        'geopot_500': data['geopot_500'],
        'humidity': data['humidity'] + humidity_adj,
        'incoming_sw': data['incoming_sw'] + incoming_sw_adj,
        'lcl_height': data['lcl_height'],
        'net_radiation': data['net_radiation'],
        'omega_500': data['omega_500'],
        'pbl_height': data['pbl_height'],
        'precipitable_water': data['precipitable_water'],
        'radiation_ratio': data['radiation_ratio'],
        'skin_temp': data['skin_temp'] + skin_temp_adj,
        'slope': data['slope'],
        'surface_pressure': data['surface_pressure'],
        'temp_2m': data['temp_2m'] + temp_2m_adj,
        'temp_850': data['temp_850'],
        'temp_dew': data['temp_dew'] + temp_dew_adj,
        'temp_wet': data['temp_wet']
    })
    # Scale and predict
    return pipeline.predict(X_adj)

# Apply predictions
sim_data['UHI_index'] = predict_UHI(sim_data)

# Create map
bounds = sim_data.total_bounds
buffer = 0.05
map = folium.Map(
    location=[8.48, 124.65],
    zoom_start=11,
    min_zoom=11,
    max_zoom=18,
    tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
    ),
    max_bounds=True,
    min_lat=bounds[1]-buffer,
    max_lat=bounds[3]+buffer,
    min_lon=bounds[0]-buffer,
    max_lon=bounds[2]+buffer,
    control_scale=False
)

# Set visualization range
vmin, vmax = 0, 5
sim_data['UHI_vis'] = sim_data['UHI_index'].clip(vmin, vmax)

# Add choropleth layer (using simulated UHI values)
folium.Choropleth(
    geo_data=sim_data,
    data=sim_data,
    name="UHI Intensity",
    columns=["barangay", "UHI_vis"],
    key_on="feature.properties.barangay",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="UHI Intensity (°C)",
    bins=[0, 1, 2, 3, 4, 5]
).add_to(map)

# Add interactive tooltips
folium.GeoJson(
    sim_data,
    style_function=lambda x: {'color': 'black', 'weight': 0.5, 'fillOpacity': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["barangay", "UHI_index"],
        aliases=["Barangay:", "UHI Intensity:"],
        localize=True,
        style=("font-weight: bold;")
    )
).add_to(map)

# Full-page map display
st_folium(map, use_container_width=True, height=820)
