import streamlit as st
import pandas as pd
import geopandas as gpd
import joblib
import folium
import leafmap.foliumap as leafmap
import branca.colormap as cm

# CONFIGURE PAGE LAYOUT
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
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

# [STREAMLIT] METRIC VALUE SIZE
metric_value = """
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 800;
}
</style>
"""
st.markdown(metric_value, unsafe_allow_html=True)

# LOAD FUNCTIONS (CACHED FOR PERFORMANCE)
@st.cache_resource
def load_data():
    cdo = gpd.read_parquet("cdo_geodata.parquet")
    features = pd.read_csv("latest_data.csv")
    info = pd.read_csv("info.csv", index_col=0)
    return cdo.to_crs(epsg=4326), features, info

@st.cache_resource 
def load_model():
    return joblib.load("model.joblib")

# LOAD ALL DATA
cdo_gdf, features_df, info_df = load_data()
model = load_model()

# MERGE FEATURES WITH GEOMETRIES
sim_data = cdo_gdf.merge(features_df, on='barangay')
sim_data['barangay'] = sim_data['barangay'].str.replace(r'\bBarangay\b', 'Brgy', regex=True)

# MULTIPLIER SLIDER PARAMETERS
multiplier_min = 0.1
multiplier_max = 2.0
multiplier_default = 1.0
multiplier_step = 0.05

with st.sidebar:
    st.title("🎛️ Simulation Controls", help="**How it works:** Adjust the sliders to simulate changes in the environment. Each slider uses a multiplier to increase or decrease the influence of a specific factor on Urban Heat. A value of 1.0 means no change, while lower or higher values represent less or more impact.")
    
    with st.expander("🏙️ Urban Surface Features", expanded=True):
        ndbi_mult = st.slider(
            "More or Less Buildings",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Built-up index intensity (Normalized Difference Built-up Index)."
        )
        nlights_mult = st.slider(
            "More or Less Artificial Lights",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Nighttime light intensity from satellite observations."
        )
    
    with st.expander("🌬️ Atmospheric Conditions", expanded=False):
        omega_mult = st.slider(
            "Up or Down Air Movement",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Vertical atmospheric motion (omega at 500 hPa level)."
        )
    
    with st.expander("🌿 Cooling & Environment", expanded=False):
        cooling_mult = st.slider(
            "More or Less Cooling Areas",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Cooling potential of green spaces and heat-absorbing surfaces."
        )
        canyon_mult = st.slider(
            "More or Less Trapped Heat Between Buildings",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="The urban canyon effect from dense, tall buildings."
        )
        micro_mult = st.slider(
            "Local Shade or Moisture Changes",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Microclimate modifiers like shading and surface moisture."
        )
        dtr_mult = st.slider(
            "Bigger or Smaller Day-Night Temp Gaps",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Diurnal temperature range (difference between day and night temps)."
        )

# APPLY MULTIPLIERS TO FEATURE VALUES
sim_scaled = sim_data.copy()
ndbi_min, ndbi_max = info_df.loc['min', 'NDBI'], info_df.loc['max', 'NDBI']
nlights_min, nlights_max = info_df.loc['min', 'nighttime_lights'], info_df.loc['max', 'nighttime_lights']
omega_min, omega_max = info_df.loc['min', 'omega_500'], info_df.loc['max', 'omega_500']
cooling_min, cooling_max = info_df.loc['min', 'cooling_capacity'], info_df.loc['max', 'cooling_capacity']
canyon_min, canyon_max = info_df.loc['min', 'canyon_effect'], info_df.loc['max', 'canyon_effect']
micro_min, micro_max = info_df.loc['min', 'microclimate_mod'], info_df.loc['max', 'microclimate_mod']
dtr_min, dtr_max = info_df.loc['min', 'dtr_proxy'], info_df.loc['max', 'dtr_proxy']

sim_scaled['NDBI_norm'] = (sim_scaled['NDBI'] - ndbi_min) / (ndbi_max - ndbi_min)
sim_scaled['NDBI_norm'] *= ndbi_mult
sim_scaled['NDBI'] = sim_scaled['NDBI_norm'] * (ndbi_max - ndbi_min) + ndbi_min

sim_scaled['nighttime_lights_norm'] = (sim_scaled['nighttime_lights'] - nlights_min) / (nlights_max - nlights_min)
sim_scaled['nighttime_lights_norm'] *= nlights_mult
sim_scaled['nighttime_lights'] = sim_scaled['nighttime_lights_norm'] * (nlights_max - nlights_min) + nlights_min

sim_scaled['omega_500_norm'] = (sim_scaled['omega_500'] - omega_min) / (omega_max - omega_min)
sim_scaled['omega_500_norm'] *= omega_mult
sim_scaled['omega_500'] = sim_scaled['omega_500_norm'] * (omega_max - omega_min) + omega_min

sim_scaled['cooling_capacity_norm'] = (sim_scaled['cooling_capacity'] - cooling_min) / (cooling_max - cooling_min)
sim_scaled['cooling_capacity_norm'] *= cooling_mult
sim_scaled['cooling_capacity'] = sim_scaled['cooling_capacity_norm'] * (cooling_max - cooling_min) + cooling_min

sim_scaled['canyon_effect_norm'] = (sim_scaled['canyon_effect'] - canyon_min) / (canyon_max - canyon_min)
sim_scaled['canyon_effect_norm'] *= canyon_mult
sim_scaled['canyon_effect'] = sim_scaled['canyon_effect_norm'] * (canyon_max - canyon_min) + canyon_min

sim_scaled['microclimate_mod_norm'] = (sim_scaled['microclimate_mod'] - micro_min) / (micro_max - micro_min)
sim_scaled['microclimate_mod_norm'] *= micro_mult
sim_scaled['microclimate_mod'] = sim_scaled['microclimate_mod_norm'] * (micro_max - micro_min) + micro_min

sim_scaled['dtr_proxy_norm'] = (sim_scaled['dtr_proxy'] - dtr_min) / (dtr_max - dtr_min)
sim_scaled['dtr_proxy_norm'] *= dtr_mult
sim_scaled['dtr_proxy'] = sim_scaled['dtr_proxy_norm'] * (dtr_max - dtr_min) + dtr_min

# APPLY PREDICTIONS
model_features = ['NDBI', 'nighttime_lights', 'omega_500',
                  'cooling_capacity', 'canyon_effect',
                  'microclimate_mod', 'dtr_proxy']
sim_data['UHI_index'] = model.predict(sim_scaled[model_features]).round(3)

# SET VISUALIZATION RANGE
vmin, vmax = 0, 5
sim_data['UHI_vis'] = sim_data['UHI_index'].clip(vmin, vmax)

# DASHBOARD SECTIONS
col1, col2 = st.columns(2)

# MAP SECTION
with col1:
    bounds = sim_data.total_bounds
    buffer = 0.05
    map = leafmap.Map(
        location=[8.48, 124.65],
        zoom_start=10,
        min_zoom=10,
        max_zoom=18,
        tiles="CartoDB.PositronNoLabels",
        max_bounds=True,
        min_lat=bounds[1]-buffer,
        max_lat=bounds[3]+buffer,
        min_lon=bounds[0]-buffer,
        max_lon=bounds[2]+buffer,
        control_scale=False,
        search_control=False,
        layer_control=False,
    )

    # CONSISTENT COLOR DEFINITION
    bins = [0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5]
    colors = ['#ffffff', '#ffffd9', '#fff5b8', '#ffde82', '#ffc04d', '#ff9a2e', '#ff6b39', '#cc0000']
    
    # FUNCTION TO CREATE COLOR MAPPING
    def get_color(value):
        for i, b in enumerate(bins[1:]):
            if value < b:
                return colors[i]
        return colors[-1]
    
    # CREATE CUSTOM COLORMAP FOR THE CHOROPLETH
    colormap = cm.LinearColormap(
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        caption="UHI Intensity (°C)"
    )

    # STYLE FUNCTION FOR GEOJSON
    def style_function(feature):
        value = feature['properties']['UHI_vis']
        return {
            'fillColor': get_color(value),
            'fillOpacity': 0.75,
            'color': 'black',
            'weight': 1.0,
            'opacity': 0.9
        }

    # ADD GEOJSON LAYER
    folium.GeoJson(
        data=sim_data,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["barangay", "UHI_index"],
            aliases=["Barangay:", "UHI Intensity (°C):"],
            style=("font-weight: bold; font-size: 12px;"),
            sticky=True
        ),
        name="UHI Intensity",
    ).add_to(map)

    # ADD THE COLOR LEGEND
    colormap.add_to(map)

    # DISPLAY MAP
    st.subheader("🗺️ UHI Distribution Map")
    map.to_streamlit(height=580, width=None, add_layer_control=False)

# TABLE AND METRIC SECTION
with col2:
    all_barangays = sim_data[['barangay', 'UHI_index']].sort_values(by='UHI_index', ascending=False).reset_index(drop=True)

    # FORMAT THE DATAFRAME
    display_df = all_barangays.copy()
    display_df['UHI_index'] = display_df['UHI_index'].map(lambda x: f"{x:.3f}")
    display_df = display_df.rename(columns={"barangay":"Barangay", "UHI_index":"UHI Index"})
    
    # FUNCTION TO STYLE EACH CELL BASED ON UHI VALUE
    def color_uhi_values(val):
        if not isinstance(val, str):
            return ''
        try:
            val_float = float(val)
            color = get_color(val_float)
            dark_colors = ['#ff6b39', '#cc0000']
            text_color = "white" if color in dark_colors else "black"
            return f'background-color: {color}; color: {text_color}; font-weight: bold'
        except:
            return ''
    
    # APPLY STYLING TO UHI INDEX COLUMN
    styled_table = display_df.style.applymap(color_uhi_values, subset=['UHI Index'])
    
    # DISPLAY THE TABLE
    st.subheader("📍Barangays by UHI Intensity")
    st.dataframe(styled_table, height=400, use_container_width=True)

    # CALCULATE SUMMARY METRICS
    avg_uhi = sim_data['UHI_index'].mean().round(3)
    hottest_barangay = sim_data.loc[sim_data['UHI_index'].idxmax()]
    coolest_barangay = sim_data.loc[sim_data['UHI_index'].idxmin()]

    # DISPLAY THE METRICS
    st.subheader("🔍 Summary Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Average UHI Index", f"{avg_uhi:.3f} °C", border=True)
    m2.metric(f"Hottest Barangay ({hottest_barangay['UHI_index']:.3f} °C)", hottest_barangay['barangay'], border=True)
    m3.metric(f"Coolest Barangay ({coolest_barangay['UHI_index']:.3f} °C)", coolest_barangay['barangay'], border=True)
