import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import folium
import leafmap.foliumap as leafmap

# CONFIGURE PAGE LAYOUT
st.set_page_config(layout="wide")
st.logo(image="logo.png", size="large")

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

# [LEAFMAP] ADD MAP BORDER
map_border_style = """
<style>
iframe {
    border: 1px solid white !important;
    box-sizing: border-box;
}
</style>
"""
st.markdown(map_border_style, unsafe_allow_html=True)

# LOAD FUNCTIONS (CACHED FOR PERFORMANCE)
@st.cache_resource
def load_data():
    cdo = gpd.read_parquet("cdo_geodata.parquet")
    features = pd.read_csv("latest_data.csv")
    return cdo.to_crs(epsg=4326), features

@st.cache_resource 
def load_model():
    return joblib.load("model.joblib")

# LOAD ALL DATA
cdo_gdf, features_df = load_data()
model = load_model()

# MERGE FEATURES WITH GEOMETRIES
sim_data = cdo_gdf.merge(features_df, on='barangay')

# SIMULATION CONTROLS
with st.sidebar:
    st.title("Simulation Controls")
    st.markdown("Adjust each feature to simulate UHI changes using multipliers.")
    
    with st.expander("🏙️ Urban Form"):
        ndbi_mult = st.slider(
            "Urban Density Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Normalized Difference Built-up Index – higher values mean more urban surfaces."
        )
        lights_mult = st.slider(
            "Nighttime Lights Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Brightness at night – proxy for human activity and infrastructure."
        )
        canyon_mult = st.slider(
            "Urban Canyon Effect Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Buildings trap heat – higher values mean stronger canyon effect."
        )
        
    with st.expander("🌡️ Heat & Airflow"):
        omega_mult = st.slider(
            "Air Motion (500 hPa) Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Vertical air motion – influences heat dispersion and cloud formation."
        )
        cooling_mult = st.slider(
            "Cooling Capacity Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Natural ability of area to cool itself – higher values reduce UHI."
        )
        dtr_mult = st.slider(
            "Diurnal Temp Range Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Temperature difference between day and night – relates to heat retention."
        )
        
    with st.expander("🌿 Microclimate"):
        microclimate_mult = st.slider(
            "Microclimate Modifier Multiplier", 0.5, 1.5, 1.0, 0.01,
            help="Land cover's impact on local temperature, humidity, and wind."
        )
        
# PREDICTION FUNCTION
def predict_UHI(data):
    X_adj = data.copy()
    X_adj = X_adj.drop(columns=['barangay', 'geometry'])
    X_adj['NDBI'] = X_adj['NDBI'] * ndbi_mult
    X_adj['nighttime_lights'] = X_adj['nighttime_lights'] * lights_mult
    X_adj['omega_500'] = X_adj['omega_500'] * omega_mult
    X_adj['cooling_capacity'] = X_adj['cooling_capacity'] * cooling_mult
    X_adj['canyon_effect'] = X_adj['canyon_effect'] * canyon_mult
    X_adj['microclimate_mod'] = X_adj['microclimate_mod'] * microclimate_mult
    X_adj['dtr_proxy'] = X_adj['dtr_proxy'] * dtr_mult
    return model.predict(X_adj)

# APPLY PREDICTIONS
sim_data['UHI_index'] = predict_UHI(sim_data)
sim_data['UHI_index'] = sim_data['UHI_index'].round(3)

# CREATE MAP
bounds = sim_data.total_bounds
buffer = 0.05
map = leafmap.Map(
    location=[8.48, 124.65],
    zoom_start=11,
    min_zoom=11,
    max_zoom=18,
    tiles="CartoDB.PositronNoLabels",
    max_bounds=True,
    min_lat=bounds[1]-buffer,
    max_lat=bounds[3]+buffer,
    min_lon=bounds[0]-buffer,
    max_lon=bounds[2]+buffer,
    search_control=False,
)

# SET VISUALIZATION RANGE
vmin, vmax = 0, 5
sim_data['UHI_vis'] = sim_data['UHI_index'].clip(vmin, vmax)

# ADD CHOROPLETH LAYER
folium.Choropleth(
    geo_data=sim_data.__geo_interface__,
    data=sim_data,
    columns=["barangay", "UHI_vis"],
    key_on="feature.properties.barangay",
    fill_color="YlOrRd",
    fill_opacity=0.5,
    line_opacity=0,
    legend_name="UHI Intensity (°C)",
    bins=[0, 1, 2, 3, 4, 5],
    name="UHI Intensity",
    control=False
).add_to(map)

# ADD TOOLTIPS
folium.GeoJson(
    data=sim_data,
    style_function=lambda x: {'color': 'black', 'weight': 0.5, 'fillOpacity': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["barangay", "UHI_index"],
        aliases=["Barangay:", "UHI Intensity (°C):"],
        style=("font-weight: bold; font-size: 12px;"),
        sticky=True
    ),
    name="Tooltips",
    control=False
).add_to(map)

# DISPLAY MAP
map.to_streamlit(use_container_width=True)
