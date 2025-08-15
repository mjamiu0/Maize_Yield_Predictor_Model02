import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets with error handling
try:
    loaded_model = joblib.load('Mazie_yield_prediction_models02.pkl')
    columns = joblib.load('model_columns02.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Feature configuration
soil_columns = [col for col in columns if 'SOIL TYPE PERCENT1' in col]
soil_types = [col.replace('SOIL TYPE PERCENT1 (Percent)_', '') for col in soil_columns]

state_columns = [col for col in columns if 'State Name' in col]
state_names = [col.replace('State Name_', '') for col in state_columns]

dist_columns = [col for col in columns if 'Dist Name' in col]
dist_names = [col.replace('Dist Name_', '') for col in dist_columns]

numerical_features = [
    'NITROGEN PER HA OF GCA (Kg per ha)',
    'PHOSPHATE PER HA OF GCA (Kg per ha)',
    'POTASH PER HA OF GCA (Kg per ha)',
    'AVERAGE RAINFALL (Millimeters)',   
    'AVERAGE RAINFALL (Millimeters)',
    'AVERAGE TEMPERATURE (Centigrate)',
    'AVERAGE PERCIPITATION (Millimeters)',
    'Year'
]

# Ensure we're using the exact column names from the trained model
features = columns  # Use the loaded columns directly

# Streamlit UI
st.title('Maize Yield Prediction Model')
st.markdown("### Predict the maize yield based on environmental and farming factors.")

# Input widgets
st.header('Input Parameters')
year = st.slider('Year', min_value=1966, max_value=2025, value=2023)
avg_temp = st.number_input('Average Temperature (Centigrate)', value=25.0)
nitrogen = st.number_input('Nitrogen per ha of GCA (Kg per ha)', value=10.0)
phosphate = st.number_input('Phosphate per ha of GCA (Kg per ha)', value=5.0)
potash = st.number_input('Potash per ha of GCA (Kg per ha)', value=3.0)
rainfall = st.number_input('Average Rainfall (Millimeters)', value=150.0)
precipitation = st.number_input('Average Precipitation (Millimeters)', value=100.0)
selected_soil_type = st.selectbox('Soil Type', soil_types)
selected_state_name = st.selectbox('State Name', state_names)
selected_dist_name = st.selectbox('District Name', dist_names)

if st.button('Predict Maize Yield'):
    # Create input dictionary with all features initialized to 0
    input_data = {col: 0 for col in features