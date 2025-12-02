import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import requests

st.set_page_config(page_title="Metro Manila Flood Risk", layout="wide")

# ----------------------------------
# Title
# ----------------------------------
st.title("🌧️ Metro Manila Flood Risk Map")
st.write("This map shows possible flood-prone locations around Metro Manila using simple risk categories.")

# ----------------------------------
# Metro Manila GeoJSON (Boundary Outline)
# ----------------------------------
# Free public GeoJSON of NCR
GEOJSON_URL = "https://raw.githubusercontent.com/gavinr/world-countries-centroids/master/data/philippines/ncr.geojson"

try:
    response = requests.get(GEOJSON_URL)
    metro_manila_geojson = response.json()
except:
    st.error("Failed to load Metro Manila map boundary.")
    metro_manila_geojson = None

# ----------------------------------
# Sample Flood Risk Data (Example)
# ----------------------------------
flood_risk_data = [
    {"name": "Marikina River", "lat": 14.6508, "lon": 121.1023, "risk": "High"},
    {"name": "Malabon Area", "lat": 14.6688, "lon": 120.9657, "risk": "High"},
    {"name": "España Blvd, Manila", "lat": 14.6100, "lon": 120.9890, "risk": "Medium"},
    {"name": "Quezon City Circle", "lat": 14.6538, "lon": 121.0480, "risk": "Low"},
]

risk_colors = {
    "Low": "green",
    "Medium": "orange",
    "High": "red"
}

# ----------------------------------
# Create Map
# ----------------------------------
m = folium.Map(location=[14.5995, 120.9842], zoom_start=11)

# Add Metro Manila Boundary
if metro_manila_geojson:
    folium.GeoJson(
        metro_manila_geojson,
        name="Metro Manila Boundary",
        style_function=lambda x: {
            "fillColor": "none",
            "color": "blue",
            "weight": 2,
        }
    ).add_to(m)

# Add Flood Risk Markers
for item in flood_risk_data:
    folium.CircleMarker(
        location=[item["lat"], item["lon"]],
        radius=8,
        color=risk_colors[item["risk"]],
        fill=True,
        fill_opacity=0.7,
        popup=f"{item['name']} - {item['risk']} Risk",
    ).add_to(m)

# Add Layer Control
folium.LayerControl().add_to(m)

# Display Map
st_folium(m, width=1100, height=600)

# ----------------------------------
# Legend
# ----------------------------------
st.markdown("""
### 🟩 Legend
- **🟩 Green:** Low Flood Risk  
- **🟧 Orange:** Medium Flood Risk  
- **🟥 Red:** High Flood Risk  
""")
