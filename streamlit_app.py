# Metro Manila Flood Risk - Streamlit App
# Filename: metro_manila_flood_risk_app.py
# Requirements (pip):
# streamlit, pandas, numpy, pydeck
# Optional (for improved map interactivity): streamlit-folium, folium, geopandas
# To run: pip install streamlit pandas numpy pydeck
# then: streamlit run metro_manila_flood_risk_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from io import BytesIO

st.set_page_config(page_title="Metro Manila Flood Risk", layout="wide")

# --- Helper functions -------------------------------------------------

def generate_grid(lat_min, lat_max, lon_min, lon_max, n_lat=100, n_lon=100):
    """Generate a grid of lat/lon points inside bounding box."""
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    grid = []
    for lat in lats:
        for lon in lons:
            grid.append((lat, lon))
    return np.array(grid)


def normalize(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def distance_to_line(points, line_points):
    """Compute approximate distance from each point to a polyline defined by line_points.
    Uses minimum haversine-like approximation (assuming small distances) for speed.
    points: Nx2 array of (lat, lon)
    line_points: Mx2 array of (lat, lon)
    returns: distances in kilometers (approx)
    """
    # Convert to arrays
    P = np.array(points)
    L = np.array(line_points)
    # approximate conversion: 1 degree lat ~ 111 km, lon scaling by cos(lat)
    # We'll compute distances to each segment's endpoints and take min.
    lat_scale = 111.0
    distances = np.full(len(P), np.inf)
    for lp in L:
        dy = (P[:,0] - lp[0]) * lat_scale
        dx = (P[:,1] - lp[1]) * (lat_scale * np.cos(np.deg2rad(lp[0])))
        d = np.sqrt(dx*dx + dy*dy)
        distances = np.minimum(distances, d)
    return distances


def compute_risk(grid, rainfall, elevation, dist_to_river, weights):
    """Compute a simple risk score as weighted sum of normalized factors.
    weights: dict with keys 'rain', 'elev', 'dist', 'hist' (historical hotspots)
    """
    # normalize inputs
    r_n = normalize(rainfall)
    e_n = normalize(elevation)
    # distance: closer to river -> higher risk, so invert distance
    d_inv = 1.0 - normalize(dist_to_river)

    # Basic weighted combination
    score = (weights['rain'] * r_n +
             weights['elev'] * e_n +
             weights['dist'] * d_inv)

    # clip to 0-1
    score = np.clip(score, 0, 1)
    return score


# --- Metro Manila bounding box (approx) --------------------------------
# These coordinates cover most of Metro Manila metropolitan area (for demo)
LAT_MIN, LAT_MAX = 14.40, 14.80
LON_MIN, LON_MAX = 120.90, 121.10

# --- UI ---------------------------------------------------------------
st.title("Metro Manila Flood Risk")
st.markdown(
    """
    This interactive demo computes a **simple flood risk index** across Metro Manila.

    - You can **upload historical flood incident data** (CSV with `latitude`, `longitude`, `severity` optional)
    - Or use the **sample demo data** provided here
    - Adjust factor weights (rainfall, elevation, distance-to-river) to see how risk changes

    **Note:** This is a demo/prototype for visualization and learning purposes. For operational risk maps use validated, high-resolution datasets (rainfall radar, DEM, drainage network, land cover, historic flood layers) and accepted hydrologic/hydraulic models.
    """
)

# Sidebar controls
st.sidebar.header("Controls")
use_sample = st.sidebar.checkbox("Use demo/sample data (no upload)", value=True)
resolution = st.sidebar.slider("Grid resolution (per axis)", min_value=40, max_value=300, value=120)

st.sidebar.markdown("---")
st.sidebar.subheader("Factor weights (sum need not be 1)")
w_rain = st.sidebar.slider("Rainfall weight", 0.0, 2.0, 0.8, 0.05)
w_elev = st.sidebar.slider("Elevation weight", 0.0, 2.0, 0.6, 0.05)
w_dist = st.sidebar.slider("Distance-to-river weight", 0.0, 2.0, 0.6, 0.05)
weights = {'rain': w_rain, 'elev': w_elev, 'dist': w_dist}

st.sidebar.markdown("---")
show_incidents = st.sidebar.checkbox("Show uploaded / sample historical incidents", value=True)
show_risk_legend = st.sidebar.checkbox("Show risk legend", value=True)

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload historical flood incidents CSV (columns: latitude, longitude, severity optional)", type=["csv"])

# --- Data preparation -------------------------------------------------
if use_sample or (uploaded_file is None):
    st.sidebar.info("Using demo synthetic data — replace with real datasets by disabling demo and uploading CSVs.")

# generate grid
grid = generate_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, n_lat=resolution, n_lon=resolution)

# synthetic rainfall: make a gaussian blob around eastern area (simulate heavy rain over some areas)
# Convert lat/lon to 2D coords in 0..1
lat_norm = (grid[:,0] - LAT_MIN) / (LAT_MAX - LAT_MIN)
lon_norm = (grid[:,1] - LON_MIN) / (LON_MAX - LON_MIN)

# Simulate two rainfall centers
center1 = np.array([0.6, 0.55])
center2 = np.array([0.2, 0.4])
rainfall = np.exp(-((lat_norm - center1[0])**2 + (lon_norm - center1[1])**2) / 0.01)
rainfall += 0.9 * np.exp(-((lat_norm - center2[0])**2 + (lon_norm - center2[1])**2) / 0.02)

# synthetic elevation: higher in north-west, lower near Manila Bay
elevation = (1.0 - lon_norm) * (lat_norm * 0.7 + 0.3)
# add some speckle
elevation += 0.1 * np.random.RandomState(42).rand(len(elevation))

# synthetic river polyline (a simplified Pasig-Laguna cluster line)
river_line = np.array([
    [14.80, 120.95],
    [14.72, 121.00],
    [14.65, 121.02],
    [14.55, 121.03],
    [14.45, 121.02]
])

# compute distances to river
dist_km = distance_to_line(grid, river_line)

# if user uploaded incidents, load them; otherwise create sample incident hotspots
if uploaded_file is not None and not use_sample:
    try:
        incidents = pd.read_csv(uploaded_file)
        if not {'latitude', 'longitude'}.issubset(incidents.columns):
            st.sidebar.error("CSV must contain 'latitude' and 'longitude' columns.")
            incidents = None
        else:
            # fill severity if missing
            if 'severity' not in incidents.columns:
                incidents['severity'] = 1.0
            else:
                incidents['severity'] = incidents['severity'].fillna(1.0)
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        incidents = None
else:
    # create a few sample historical incidents
    incidents = pd.DataFrame({
        'latitude': [14.589, 14.676, 14.544, 14.616, 14.706],
        'longitude': [121.035, 120.995, 121.041, 120.985, 120.965],
        'severity': [3, 2, 4, 1, 2]
    })

# incorporate historical hotspots as a factor: increase risk near incident points
if incidents is not None:
    incident_points = incidents[['latitude', 'longitude']].values
    hist_dist = distance_to_line(grid, incident_points)
    # inverse: closer to incidents means higher historic factor
    hist_factor = 1.0 - normalize(hist_dist)
else:
    hist_factor = np.zeros(len(grid))

# compute risk
base_risk = compute_risk(grid, rainfall, elevation, dist_km, weights)
# boost risk where historical incidents exist
risk_score = np.clip(base_risk + 0.3 * hist_factor, 0, 1)

# Prepare DataFrame
df = pd.DataFrame({
    'latitude': grid[:,0],
    'longitude': grid[:,1],
    'rainfall': rainfall,
    'elevation': elevation,
    'dist_km': dist_km,
    'risk': risk_score
})

# --- Map visualization using pydeck ----------------------------------

# Color scale helper: convert 0..1 to RGB
def risk_to_color(r):
    # green (low) -> yellow -> red (high)
    r = np.clip(r, 0, 1)
    # simple interpolation
    if r < 0.5:
        # green to yellow
        t = r / 0.5
        r_c = int(255 * t)
        g_c = 255
        b_c = 0
    else:
        # yellow to red
        t = (r - 0.5) / 0.5
        r_c = 255
        g_c = int(255 * (1 - t))
        b_c = 0
    return [r_c, g_c, b_c]

# sample a subset of points for performance if grid is very large
max_points_display = 20000
if len(df) > max_points_display:
    display_df = df.sample(max_points_display, random_state=1)
else:
    display_df = df

display_df['color'] = display_df['risk'].apply(risk_to_color)

midpoint = (np.mean(display_df['latitude']), np.mean(display_df['longitude']))

# build pydeck layer for risk points
risk_layer = pdk.Layer(
    "ScatterplotLayer",
    data=display_df,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=200,  # radius in meters
    pickable=True,
    opacity=0.8,
)

layers = [risk_layer]

# add incident layer
if show_incidents and incidents is not None:
    incidents_display = incidents.copy()
    incidents_display['color'] = incidents_display.get('severity', 1).apply(lambda s: [200, 30, 30])
    incident_layer = pdk.Layer(
        "ScatterplotLayer",
        data=incidents_display,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius=400,
        pickable=True,
        opacity=0.9,
    )
    layers.append(incident_layer)

# deck
view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11, pitch=0)

deck = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=layers,
    tooltip={"text": "Risk: {risk}\nLat: {latitude}\nLon: {longitude}"}
)

# column layout: map | controls & table
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Risk map — Metro Manila (demo)")
    st.pydeck_chart(deck)
    if show_risk_legend:
        st.markdown("**Legend**: green = low risk, yellow = medium, red = high")

with col2:
    st.subheader("Risk summary & data")
    st.metric("Grid points", f"{len(df):,}")
    st.write("Weight settings:")
    st.write(pd.DataFrame({'factor': ['rain', 'elev', 'dist'], 'weight': [w_rain, w_elev, w_dist]}))

    # show top hotspots
    top = df.sort_values('risk', ascending=False).head(10).reset_index(drop=True)
    st.write("Top 10 highest-risk grid cells (demo)")
    st.dataframe(top[['latitude','longitude','risk']])

    # allow download of grid CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download risk grid CSV", data=csv, file_name='metro_manila_risk_grid.csv', mime='text/csv')

# --- Additional explanatory panels -----------------------------------
st.markdown("---")
st.header("How the demo risk is calculated")
st.markdown(
    """
    This prototype calculates a simple risk index from three synthetic / user-supplied factors:

    1. **Rainfall**: simulated spatial rainfall intensity (higher is worse)
    2. **Elevation**: simulated elevation proxy (lower elevations nearer the bay are more vulnerable)
    3. **Distance to river**: points closer to a simplified centreline (river) are more exposed

    Each factor is normalized to 0..1 and combined with user-chosen weights. Historical incidents (if uploaded) boost nearby cells.

    **Important:** this is NOT a replacement for proper hydrologic/hydraulic modelling. It is meant to be a reproducible demo you can extend with real data (DEM, rainfall radar, drainage networks, flood extents, land use, etc.).
    """
)

st.header("Next steps / ideas to improve this prototype")
st.write("""
- Replace synthetic rainfall/elevation with real datasets (e.g., CHIRPS, satellite radar rainfall, local gauge networks, or municipal DEMs).
- Use high-resolution Digital Elevation Model (DEM) and perform hydraulic flow-routing / flood inundation modeling (e.g., HEC-RAS, LISFLOOD, or deltares tools).
- Incorporate drainage network capacity and land cover (impervious area).
- Validate using historic flood extent polygons and ground observations.
- Expose model calibration tools and uncertainty quantification.
""")

# small footer
st.markdown("---")
st.caption("Demo app — not for operational decision-making. Built with Streamlit.")
