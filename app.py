import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import numpy as np

# Configure page layout
st.set_page_config(layout="wide")

# App title and sidebar
st.sidebar.title("UHI Simulation Controls")
st.title("CDO Urban Heat Island Visualization")

# Load data
@st.cache_data
def load_geojson():
    ph_barangays = gpd.read_file("PH_Adm4_BgySubMuns.zip")
    cdo_barangays = ph_barangays[ph_barangays['adm3_psgc'] == 1030500000]
    return cdo_barangays[['adm4_en', 'geometry']].rename(columns={'adm4_en': 'barangay'}).to_crs(epsg=4326)

cdo_gdf = load_geojson()

# Simulation controls in sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    urban_heat = st.slider("Urban Heat Intensity", 0.0, 5.0, 2.5, 0.1)
    vegetation = st.slider("Vegetation Coverage", 0.0, 1.0, 0.5, 0.01)
    simulation_type = st.selectbox("Scenario", ["Current", "Future Projection", "Mitigation Scenario"])
    
    # Generate random UHI values for demo (replace with your model)
    cdo_gdf['uhi'] = np.random.uniform(urban_heat-1, urban_heat+1, len(cdo_gdf)) * vegetation

# Calculate bounds with buffer
bounds = cdo_gdf.total_bounds
buffer = 0.05
m = folium.Map(
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

# Add choropleth layer (using simulated UHI values)
folium.Choropleth(
    geo_data=cdo_gdf,
    name="UHI Intensity",
    data=cdo_gdf,
    columns=["barangay", "uhi"],
    key_on="feature.properties.barangay",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="UHI Intensity (°C)",
    bins=[0, 1, 2, 3, 4, 5]
).add_to(m)

# Add interactive tooltips
folium.GeoJson(
    cdo_gdf,
    style_function=lambda x: {'color': 'black', 'weight': 0.5, 'fillOpacity': 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["barangay", "uhi"],
        aliases=["Barangay:", "UHI Intensity:"],
        localize=True,
        style=("font-weight: bold;")
    )
).add_to(m)

# Full-page map display
folium_static(m, width=1400, height=700)
