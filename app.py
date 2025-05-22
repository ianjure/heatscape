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

# GET MIN-MAX DATA
info_df = pd.read_csv("all_info.csv", index_col=0)

# SIMULATION LOGIC
level_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
level_values = [0, 0.25, 0.5, 0.75, 1.0]  # MAP TO MIN-MAX INTERPOLATION

def scale_value(feature, level):
    min_val = info_df.loc["min", feature]
    max_val = info_df.loc["max", feature]
    return min_val + (max_val - min_val) * level

with st.expander("🏙️ Urban Surface Features"):
        ndbi_level = st.select_slider("Built Environment (NDBI)", level_labels, value="Medium",
                                      help="Adjust built-up index intensity")
        nlights_level = st.select_slider("Artificial Lighting", level_labels, value="Medium",
                                         help="Adjust nighttime light intensity")

    with st.expander("🌬️ Atmospheric Conditions"):
        omega_level = st.select_slider("Vertical Air Motion (ω500)", level_labels, value="Medium",
                                       help="Adjust vertical atmospheric motion")

    with st.expander("🌿 Cooling & Environment"):
        cooling_level = st.select_slider("Cooling Potential", level_labels, value="Medium",
                                         help="Adjust cooling capacity of green spaces and surfaces")
        canyon_level = st.select_slider("Urban Canyon Effect", level_labels, value="Medium",
                                        help="Adjust building canyon trapping effect")
        micro_level = st.select_slider("Microclimate Modifier", level_labels, value="Medium",
                                       help="Adjust local modifiers like shade or humidity")
        dtr_level = st.select_slider("Day-Night Temp Range (DTR)", level_labels, value="Medium",
                                     help="Adjust the range between daytime and nighttime temperatures")

# APPLY SCALED VALUES BASED ON LEVELS
X_adj = pd.DataFrame({
    'NDBI': scale_value('NDBI', level_values[level_labels.index(ndbi_level)]),
    'nighttime_lights': scale_value('nighttime_lights', level_values[level_labels.index(nlights_level)]),
    'omega_500': scale_value('omega_500', level_values[level_labels.index(omega_level)]),
    'cooling_capacity': scale_value('cooling_capacity', level_values[level_labels.index(cooling_level)]),
    'canyon_effect': scale_value('canyon_effect', level_values[level_labels.index(canyon_level)]),
    'microclimate_mod': scale_value('microclimate_mod', level_values[level_labels.index(micro_level)]),
    'dtr_proxy': scale_value('dtr_proxy', level_values[level_labels.index(dtr_level)]),
}, index=sim_data.index)

# APPLY PREDICTIONS
X_full = pd.concat([X_adj] * len(sim_data)).reset_index(drop=True)
sim_data['UHI_index'] = model.predict(X_full).round(3)

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
