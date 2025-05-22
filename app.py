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

with st.sidebar:
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

# APPLY MULTIPLIERS TO FEATURE VALUES
sim_scaled = sim_data.copy()
sim_scaled['NDBI'] *= ndbi_mult
sim_scaled['nighttime_lights'] *= nlights_mult
sim_scaled['omega_500'] *= omega_mult
sim_scaled['cooling_capacity'] *= cooling_mult
sim_scaled['canyon_effect'] *= canyon_mult
sim_scaled['microclimate_mod'] *= micro_mult
sim_scaled['dtr_proxy'] *= dtr_mult

# APPLY PREDICTIONS
model_features = ['NDBI', 'nighttime_lights', 'omega_500',
                  'cooling_capacity', 'canyon_effect',
                  'microclimate_mod', 'dtr_proxy']
sim_data['UHI_index'] = model.predict(sim_scaled[model_features]).round(3)

col1, col2 = st.columns(2)
with col1:
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
    map.to_streamlit(width=None, height=100, add_layer_control=False)

with col2:
    # Add a table showing all barangays with scrolling capability
    st.subheader("Barangays by UHI Intensity")
    
    # Get all barangays sorted by UHI index
    all_barangays = sim_data[['barangay', 'UHI_index']].sort_values(by='UHI_index', ascending=False).reset_index(drop=True)
    
    # Format the UHI values to 3 decimal places but keep numeric version for styling
    display_df = all_barangays.copy()
    display_df['UHI_numeric'] = display_df['UHI_index']  # Keep numeric version
    display_df['UHI_index'] = display_df['UHI_index'].map(lambda x: f"{x:.3f}")  # Format for display
    
    # Define a function to style each cell based on UHI value
    def color_uhi_values(val):
        if not isinstance(val, str):
            return ''
        
        try:
            val_float = float(val)
            # Use our same color function to ensure consistent coloring
            color = get_color(val_float)
            
            # Return the styling for the cell - use white text for darker backgrounds
            dark_colors = ['#ff6b39', '#cc0000']
            text_color = "white" if color in dark_colors else "black"
            
            return f'background-color: {color}; color: {text_color}; font-weight: bold'
        except:
            return ''
    
    # Apply styling to the UHI_index column only
    styled_table = display_df.style.applymap(
        color_uhi_values, 
        subset=['UHI_index']
    )
    
    # Display the table with fixed height and scrolling
    st.dataframe(
        styled_table,
        height=400,  # Match map height
        use_container_width=True
    )
