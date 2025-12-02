import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from google.colab import drive
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, VBox, HBox, Layout
import ipywidgets as widgets
from IPython.display import display, clear_output
import os

# --- 1. SETUP: Mount Drive and Load Data ---
print("--- 1. Setup and Data Loading ---")
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DRIVE_FILE_PATH = "/content/drive/MyDrive/ElectivePIT/Flood_Prediction_NCR_Philippines.csv"
LOCAL_FILE_NAME = "Flood_Prediction_NCR_Philippines.csv"

# Copy file locally for robust access
try:
    if not os.path.exists(LOCAL_FILE_NAME):
        !cp "{DRIVE_FILE_PATH}" .
    df = pd.read_csv(LOCAL_FILE_NAME)
    print(f"✅ Data loaded successfully from {LOCAL_FILE_NAME}.")
except Exception as e:
    print(f"❌ ERROR: Could not load data. Check path: {DRIVE_FILE_PATH}. Error: {e}")
    # Stop execution if data loading fails
    raise

# --- 2. Data Preprocessing and Model Training (Detection Mode) ---

# Use Detection Mode for the simplest interactive experience (predicting based on current data)
print("\n--- 2. Preprocessing and Training Random Forest Model ---")

# Feature Engineering for simple detection mode
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])

# Features used for the interactive model (Current Day Inputs + Geographic)
features = ['Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m', 'Location_encoded']
X = df[features]
y = df['FloodOccurrence']

# --- Data Balancing (Improves demo robustness) ---
flood_indices = df[df['FloodOccurrence'] == 1].index
no_flood_indices = df[df['FloodOccurrence'] == 0].sample(n=len(flood_indices), random_state=42).index
X_balanced = X.loc[flood_indices.union(no_flood_indices)]
y_balanced = y.loc[flood_indices.union(no_flood_indices)]

# --- Scaling and Training ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Train Random Forest with optimized parameters from your previous analysis
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_scaled, y_balanced)

# Cache locations and bounds for widgets
LOCATIONS = list(le.classes_)
RAINFALL_MAX = df['Rainfall_mm'].max()
WATER_LEVEL_MAX = df['WaterLevel_m'].max()
SOIL_MOISTURE_MEAN = df['SoilMoisture_pct'].mean()
ELEVATION_MEAN = df['Elevation_m'].mean()

print("Model Training Complete. Interactive Predictor Ready!")

# --- 3. Interactive Prediction Function and Widgets ---

def predict_flood(rainfall, water_level, soil_moisture, elevation, city):
    """
    Predicts flood risk based on user inputs using the trained model.
    This function is linked to the interactive widgets.
    """

    # Clear previous output
    clear_output(wait=True)

    # 1. Encode Location
    location_encoded_val = le.transform([city])[0]

    # 2. Create Input DataFrame in the exact feature order
    input_data = pd.DataFrame([{
        "Rainfall_mm": rainfall,
        "WaterLevel_m": water_level,
        "SoilMoisture_pct": soil_moisture,
        "Elevation_m": elevation,
        "Location_encoded": location_encoded_val
    }])

    # 3. Scale Input
    input_scaled = scaler.transform(input_data)

    # 4. Predict Probability
    prob_flood = model.predict_proba(input_scaled)[0][1]
    prob_no_flood = 1 - prob_flood
    prediction = 1 if prob_flood > 0.5 else 0

    # --- Display Results ---

    # Title
    display(widgets.HTML(f"<h2>Prediction for {city}</h2>"))

    # Prediction Status
    if prediction == 1:
        status_text = f'<div style="color: red; font-size: 24px; font-weight: bold;">⚠️ FLOOD WARNING (High Risk)</div>'
    else:
        status_text = f'<div style="color: green; font-size: 24px; font-weight: bold;">✅ NO FLOOD WARNING (Low Risk)</div>'
    display(widgets.HTML(status_text))

    # Print numerical result
    print(f"Chance of Flood: {prob_flood*100:.2f}%")
    print(f"Chance of No Flood: {prob_no_flood*100:.2f}%")

    # --- Visualization (Bar Chart) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["No Flood", "Flood"]
    values = [prob_no_flood, prob_flood]
    colors = ['#4CAF50', '#F44336'] # Green, Red

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Flood vs No Flood Probability")
    ax.set_ylabel("Probability")

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom')

    plt.show()

# --- Widget Definitions ---
# City Dropdown
city_dropdown = widgets.Dropdown(
    options=LOCATIONS,
    value=LOCATIONS[0],
    description='Location:',
    disabled=False,
    layout=Layout(width='90%')
)

# Rainfall Slider
rainfall_slider = widgets.FloatSlider(
    value=50.0,
    min=0.0,
    max=max(RAINFALL_MAX, 100.0), # Ensure max is at least 100
    step=5.0,
    description='Rainfall (mm):',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width='90%')
)

# Water Level Slider
water_level_slider = widgets.FloatSlider(
    value=2.0,
    min=0.0,
    max=max(WATER_LEVEL_MAX, 5.0), # Ensure max is at least 5
    step=0.1,
    description='Water Level (m):',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width='90%')
)

# Soil Moisture Slider
soil_moisture_slider = widgets.FloatSlider(
    value=SOIL_MOISTURE_MEAN,
    min=0.0,
    max=100.0,
    step=1.0,
    description='Soil Moisture (%):',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width='90%')
)

# Elevation Input
elevation_input = widgets.IntSlider(
    value=int(ELEVATION_MEAN),
    min=1,
    max=100,
    step=1,
    description='Elevation (m):',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width='90%')
)

# --- Interactive Interface ---
# Use an interactive object to link the function to the widgets
interactive_plot = interactive(
    predict_flood,
    rainfall=rainfall_slider,
    water_level=water_level_slider,
    soil_moisture=soil_moisture_slider,
    elevation=elevation_input,
    city=city_dropdown
)

# Display the control panel and the output
controls = widgets.VBox([
    widgets.HTML("<h3>Adjust Input Parameters</h3>"),
    city_dropdown,
    rainfall_slider,
    water_level_slider,
    soil_moisture_slider,
    elevation_input
], layout=Layout(width='450px', border='solid 1px #ccc', padding='10px', margin='10px'))

# Use the full output from the interactive call
out = interactive_plot.children[-1]
out.layout = Layout(border='solid 1px #ccc', padding='10px', margin='10px')

# Display everything in a horizontal box
display(HBox([controls, out]))
