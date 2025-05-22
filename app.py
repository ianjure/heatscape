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
sim_data['barangay'] = sim_data['barangay'].str.replace(r'\bBarangay\b', 'Brgy', regex=True)

# MULTIPLIER SLIDER PARAMETERS
# Range from 0.5x (half current) to 1.5x (50% increase)
multiplier_min = 0.5
multiplier_max = 1.5
multiplier_default = 1.0
multiplier_step = 0.05

with st.sidebar:
    with st.expander("🏙️ Urban Surface Features", expanded=True):
        ndbi_mult = st.slider(
            "Built Environment (NDBI) multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust built-up index intensity."
        )
        nlights_mult = st.slider(
            "Artificial Lighting multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust nighttime light intensity."
        )
    
    with st.expander("🌬️ Atmospheric Conditions", expanded=True):
        omega_mult = st.slider(
            "Vertical Air Motion (ω500) multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust vertical atmospheric motion."
        )
    
    with st.expander("🌿 Cooling & Environment", expanded=True):
        cooling_mult = st.slider(
            "Cooling Potential multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust cooling capacity of green spaces and surfaces."
        )
        canyon_mult = st.slider(
            "Urban Canyon Effect multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust building canyon trapping effect."
        )
        micro_mult = st.slider(
            "Microclimate Modifier multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust local modifiers like shade or humidity."
        )
        dtr_mult = st.slider(
            "Day-Night Temp Range (DTR) multiplier",
            min_value=multiplier_min,
            max_value=multiplier_max,
            value=multiplier_default,
            step=multiplier_step,
            help="Adjust the range between daytime and nighttime temperatures."
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

# Set visualization range
vmin, vmax = 0, 5
sim_data['UHI_vis'] = sim_data['UHI_index'].clip(vmin, vmax)

# Define the color bins and color scale - CONSISTENT COLOR DEFINITION
bins = [0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5]  # 8 equal bins between 0 and 5
colors = ['#ffffff', '#ffffd9', '#fff5b8', '#ffde82', '#ffc04d', '#ff9a2e', '#ff6b39', '#cc0000']

# Create color mapping function for consistent coloring
def get_color(value):
    for i, b in enumerate(bins[1:]):
        if value < b:
            return colors[i]
    return colors[-1]

# Create custom colormap for the choropleth
colormap = cm.LinearColormap(
    colors=colors,
    vmin=vmin,
    vmax=vmax,
    caption="UHI Intensity (°C)"
)

# Create two columns for map and table with more space between them
col1, col2 = st.columns(2)

with col1:
    # Create UHI map
    bounds = sim_data.total_bounds
    buffer = 0.05
    map = leafmap.Map(
        location=[8.48, 124.65],
        zoom_start=10,  # Adjusted zoom for smaller map
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

    # Define style function for GeoJson to ensure colors match our scale
    def style_function(feature):
        value = feature['properties']['UHI_vis']
        return {
            'fillColor': get_color(value),
            'fillOpacity': 0.75,
            'color': 'black',
            'weight': 1.0,
            'opacity': 0.9
        }

    # Add GeoJson layer with our custom style function
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

    # Add the color legend
    colormap.add_to(map)

    # Display map
    st.subheader("🗺️ UHI Distribution Map")
    map.to_streamlit(height=595, width=None, add_layer_control=False)

with col2:
    # Get all barangays sorted by UHI index
    all_barangays = sim_data[['barangay', 'UHI_index']].sort_values(by='UHI_index', ascending=False).reset_index(drop=True)
    
    # Format the UHI values to 3 decimal places but keep numeric version for styling
    display_df = all_barangays.copy()
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
    st.subheader("📍Barangays by UHI Intensity")
    st.dataframe(
        styled_table,
        height=400,  # Match map height
        use_container_width=True
    )

    # Add summary metrics using st.metric
    avg_uhi = sim_data['UHI_index'].mean().round(3)
    hottest_barangay = sim_data.loc[sim_data['UHI_index'].idxmax()]
    coolest_barangay = sim_data.loc[sim_data['UHI_index'].idxmin()]

    st.markdown("### 🔍 Summary Metrics")
    m1, m2, m3 = st.columns(3)

    m1.metric("Average UHI Index", f"{avg_uhi:.3f} °C", border=True)
    m2.metric(f"Hottest Barangay ({hottest_barangay['UHI_index']:.3f} °C)", hottest_barangay['barangay'], border=True)
    m3.metric(f"Coolest Barangay ({coolest_barangay['UHI_index']:.3f} °C)", coolest_barangay['barangay'], border=True)
