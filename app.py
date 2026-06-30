import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration & Styling
st.set_page_config(page_title="PLGA Formulation Dashboard", layout="centered")
st.title("🧪 PLGA Nanoparticle Property Predictor")
st.write("Adjust formulation parameters below to instantly predict particle properties across your entire ML suite.")

# 2. Load ALL 6 saved models securely using caching
@st.cache_resource
def load_all_assets():
    models = {
        "Stacking Ensemble": joblib.load('stack_model.pkl'),
        "Random Forest Regressor": joblib.load('rf_model.pkl'),
        "Gradient Boosting Regressor": joblib.load('gbr_model.pkl'),
        "Extra Trees Regressor": joblib.load('etr_model.pkl'),
        "Support Vector Regressor": joblib.load('svr_model.pkl'),
        "XGBoost Regressor": joblib.load('xgb_model.pkl')
    }
    return models

try:
    all_models = load_all_assets()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing model file! Error details: {e}")
    st.info("Make sure you have successfully exported all .pkl files from your notebook.")
    st.stop()

# 3. Dropdown Menu for Model Selection
st.subheader("🤖 Model Configuration")
selected_model_name = st.selectbox(
    "Choose the ML Architecture for Prediction:",
    options=list(all_models.keys())
)

# Pull the specific active model framework out of our dictionary
active_model = all_models[selected_model_name]

# 4. Sidebar Layout for Formulation Inputs
st.sidebar.header("Formulation Design Parameters")

polymer_MW = st.sidebar.slider("Polymer MW (Da)", min_value=5000, max_value=150000, value=50000, step=5000)
la_ga = st.sidebar.slider("LA/GA Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
mol_MW = st.sidebar.number_input("Drug Molecular Weight (g/mol)", value=500.0)
mol_logP = st.sidebar.slider("Drug LogP", min_value=-5.0, max_value=7.0, value=1.5, step=0.1)
mol_TPSA = st.sidebar.number_input("Drug TPSA (Å²)", value=90.0)
mol_melting_point = st.sidebar.number_input("Melting Point (°C)", value=150.0)
mol_Hacceptors = st.sidebar.slider("H-Bond Acceptors", 0, 15, 4)
mol_Hdonors = st.sidebar.slider("H-Bond Donors", 0, 10, 2)
mol_heteroatoms = st.sidebar.slider("Heteroatoms Count", 0, 20, 5)
drug_polymer = st.sidebar.slider("Drug/Polymer Ratio", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
surfactant_conc = st.sidebar.slider("Surfactant Concentration (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
surfactant_HLB = st.sidebar.slider("Surfactant HLB Value", min_value=1.0, max_value=20.0, value=15.0, step=0.5)
aqueous_organic = st.sidebar.slider("Aqueous/Organic Ratio", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
pH = st.sidebar.slider("pH Level", min_value=1.0, max_value=14.0, value=7.4, step=0.1)
solvent_polarity = st.sidebar.slider("Solvent Polarity Index", min_value=0.0, max_value=10.0, value=4.4, step=0.1)

# 5. Process Inputs & Run Prediction
input_data = np.array([[
    polymer_MW, la_ga, mol_MW, mol_logP, mol_TPSA, mol_melting_point,
    mol_Hacceptors, mol_Hdonors, mol_heteroatoms, drug_polymer,
    surfactant_conc, surfactant_HLB, aqueous_organic, pH, solvent_polarity
]])

feature_cols = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
                'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index']
input_df = pd.DataFrame(input_data, columns=feature_cols)

# Run prediction through whichever model backend is currently chosen
predicted_val = active_model.predict(input_df)[0]

# 6. Display the Visual Outputs
st.write("---")
st.subheader(f"🎯 Model Prediction Output ({selected_model_name})")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Particle Size", value=f"{predicted_val:.2f} nm")

if predicted_val < 200:
    st.success("✅ Ideal sizing for intracellular nano-delivery targets (< 200 nm)")
else:
    st.warning("⚠️ High particle size. Adjust surfactant ratios or polymer concentration.")