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
    st.title("🧪 Simulate UHI Factors")

    with st.expander("🏙️ Urban Surface Features"):
        ndbi_multiplier = st.slider("Built Environment (NDBI)", 0.5, 1.5, 1.0, 0.01,
                                    help="Scales the built-up index of each barangay")
        night_lights_multiplier = st.slider("Artificial Lighting", 0.5, 1.5, 1.0, 0.01,
                                           help="Scales nighttime light intensity")

    with st.expander("🌬️ Atmospheric Conditions"):
        omega_500_multiplier = st.slider("Vertical Air Motion (ω500)", 0.5, 1.5, 1.0, 0.01,
                                        help="Scales vertical motion values at 500hPa")

    with st.expander("🌿 Cooling & Environment"):
        cooling_capacity_multiplier = st.slider("Cooling Potential", 0.5, 1.5, 1.0, 0.01,
                                               help="Scales the potential cooling capacity")
        canyon_effect_multiplier = st.slider("Urban Canyon Effect", 0.5, 1.5, 1.0, 0.01,
                                            help="Scales building structure and canyon effects")
        microclimate_mod_multiplier = st.slider("Microclimate Modifier", 0.5, 1.5, 1.0, 0.01,
                                               help="Scales the modifier for local microclimates")
        dtr_proxy_multiplier = st.slider("Day-Night Temp Range (DTR)", 0.5, 1.5, 1.0, 0.01,
                                        help="Scales the range between day and night temps")
        
# PREDICTION FUNCTION
def predict_UHI(data):
    X_adj = pd.DataFrame({
        'NDBI': data['NDBI'] * ndbi_multiplier,
        'nighttime_lights': data['nighttime_lights'] * night_lights_multiplier,
        'omega_500': data['omega_500'] * omega_500_multiplier,
        'cooling_capacity': data['cooling_capacity'] * cooling_capacity_multiplier,
        'canyon_effect': data['canyon_effect'] * canyon_effect_multiplier,
        'microclimate_mod': data['microclimate_mod'] * microclimate_mod_multiplier,
        'dtr_proxy': data['dtr_proxy'] * dtr_proxy_multiplier,
    })
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
