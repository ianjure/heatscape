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

# MULTIPLIER SLIDER PARAMETERS
# Range from 0.5x (half current) to 1.5x (50% increase)
multiplier_min = 0.5
multiplier_max = 1.5
multiplier_default = 1.0
multiplier_step = 0.05

with st.expander("🏙️ Urban Surface Features"):
    ndbi_mult = st.slider(
        "Built Environment (NDBI) multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust built-up index intensity as a multiplier of current value"
    )
    nlights_mult = st.slider(
        "Artificial Lighting multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust nighttime light intensity as a multiplier of current value"
    )

with st.expander("🌬️ Atmospheric Conditions"):
    omega_mult = st.slider(
        "Vertical Air Motion (ω500) multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust vertical atmospheric motion as a multiplier of current value"
    )

with st.expander("🌿 Cooling & Environment"):
    cooling_mult = st.slider(
        "Cooling Potential multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust cooling capacity of green spaces and surfaces as a multiplier of current value"
    )
    canyon_mult = st.slider(
        "Urban Canyon Effect multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust building canyon trapping effect as a multiplier of current value"
    )
    micro_mult = st.slider(
        "Microclimate Modifier multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust local modifiers like shade or humidity as a multiplier of current value"
    )
    dtr_mult = st.slider(
        "Day-Night Temp Range (DTR) multiplier",
        min_value=multiplier_min,
        max_value=multiplier_max,
        value=multiplier_default,
        step=multiplier_step,
        help="Adjust the range between daytime and nighttime temperatures as a multiplier of current value"
    )

# APPLY MULTIPLIERS TO CURRENT FEATURE VALUES PER BARANGAY
X_adj = pd.DataFrame({
    'NDBI': sim_data['NDBI'] * ndbi_mult,
    'nighttime_lights': sim_data['nighttime_lights'] * nlights_mult,
    'omega_500': sim_data['omega_500'] * omega_mult,
    'cooling_capacity': sim_data['cooling_capacity'] * cooling_mult,
    'canyon_effect': sim_data['canyon_effect'] * canyon_mult,
    'microclimate_mod': sim_data['microclimate_mod'] * micro_mult,
    'dtr_proxy': sim_data['dtr_proxy'] * dtr_mult,
}, index=sim_data.index)

# APPLY PREDICTIONS
model_inputs = X_adj.reset_index(drop=True)
sim_data['UHI_index'] = model.predict(model_inputs).round(3)

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
sim_data['UHI_vis'] = sim_data['UHI_index'].clip(0, 5)

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
