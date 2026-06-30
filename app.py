import streamlit as st
import pandas as pd
import numpy as np
import joblib
from bayes_opt import BayesianOptimization

# 1. Page Configuration
st.set_page_config(page_title="PLGA Formulation & Design Hub", layout="centered")
st.title("🧪 PLGA Nanoparticle AI Design Suite")
st.write("Predict particle properties or inverse-design completely new formulations using Bayesian Optimization.")

# 2. Load Assets Securly
@st.cache_resource
def load_all_assets():
    models = {
        "Stacking Ensemble (Recommended)": joblib.load('stack_model.pkl'),
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
    st.error(f"⚠️ Missing model file! {e}")
    st.stop()

# 3. Create Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Property Predictor", "🧬 Bayesian Formulation Designer"])

# Define structural columns globally to ensure consistency
feature_cols = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
                'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index']

# =====================================================================
# TAB 1: FORWARD PREDICTION (YOUR EXISTING SYSTEM)
# =====================================================================
with tab1:
    st.subheader("🤖 Model Configuration")
    selected_model_name = st.selectbox("Choose the ML Architecture:", options=list(all_models.keys()), key="predict_model")
    active_model = all_models[selected_model_name]

    st.write("---")
    st.subheader("Move Sliders to Test a Formulation")
    
    # 15 Parameter Sliders
    p_mw = st.slider("Polymer MW (Da)", 5000, 150000, 50000, 5000)
    l_g = st.slider("LA/GA Ratio", 0.0, 1.0, 0.5, 0.05)
    m_mw = st.number_input("Drug Molecular Weight (g/mol)", value=500.0, key="p_m_mw")
    m_logp = st.slider("Drug LogP", -5.0, 7.0, 1.5, 0.1, key="p_m_logp")
    m_tpsa = st.number_input("Drug TPSA (Å²)", value=90.0, key="p_m_tpsa")
    m_mp = st.number_input("Melting Point (°C)", value=150.0, key="p_m_mp")
    m_ha = st.slider("H-Bond Acceptors", 0, 15, 4, key="p_m_ha")
    m_hd = st.slider("H-Bond Donors", 0, 10, 2, key="p_m_hd")
    m_het = st.slider("Heteroatoms Count", 0, 20, 5, key="p_m_het")
    d_p = st.slider("Drug/Polymer Ratio", 0.01, 0.5, 0.1, 0.01)
    s_c = st.slider("Surfactant Concentration (%)", 0.1, 5.0, 1.0, 0.1)
    s_hlb = st.slider("Surfactant HLB Value", 1.0, 20.0, 15.0, 0.5)
    a_o = st.slider("Aqueous/Organic Ratio", 1.0, 20.0, 5.0, 0.5)
    ph_val = st.slider("pH Level", 1.0, 14.0, 7.4, 0.1)
    s_pol = st.slider("Solvent Polarity Index", 0.0, 10.0, 4.4, 0.1)

    input_df = pd.DataFrame([[p_mw, l_g, m_mw, m_logp, m_tpsa, m_mp, m_ha, m_hd, m_het, d_p, s_c, s_hlb, a_o, ph_val, s_pol]], columns=feature_cols)
    predicted_val = active_model.predict(input_df)[0]

    st.metric(label="Predicted Particle Size", value=f"{predicted_val:.2f} nm")

# =====================================================================
# TAB 2: BAYESIAN INVERSE DESIGN (THE NEW ENGINE)
# =====================================================================
with tab2:
    st.subheader("🎯 Inverse-Design a Brand New Formulation")
    st.write("Specify your chemical constraints and target particle size. The Bayesian engine will search the multi-dimensional space to find the optimal recipe.")

    opt_model_name = st.selectbox("Choose baseline model for optimization evaluation:", options=list(all_models.keys()), key="opt_model")
    target_size = st.number_input("Enter Target Particle Size (nm):", min_value=50.0, max_value=500.0, value=120.0)

    st.markdown("#### 🔒 Fixed Molecule Properties\n*(These belong to the specific drug payload you want to encapsulate)*")
    opt_m_mw = st.number_input("Drug Molecular Weight (g/mol)", value=500.0, key="o_m_mw")
    opt_m_logp = st.number_input("Drug LogP", value=1.5, key="o_m_logp")
    opt_m_tpsa = st.number_input("Drug TPSA (Å²)", value=90.0, key="o_m_tpsa")
    opt_m_mp = st.number_input("Melting Point (°C)", value=150.0, key="o_m_mp")
    opt_m_ha = st.number_input("H-Bond Acceptors", value=4, key="o_m_ha")
    opt_m_hd = st.number_input("H-Bond Donors", value=2, key="o_m_hd")
    opt_m_het = st.number_input("Heteroatoms Count", value=5, key="o_m_het")

    if st.button("🚀 Run Bayesian Formulation Generator"):
        with st.spinner("Bayesian engine exploring search boundaries..."):
            eval_model = all_models[opt_model_name]

            # Define the objective function. Bayesian Optimization MAXIMIZES things. 
            # To get close to a target, we calculate the negative absolute error. Maximize negative error = error goes to zero!
            def optimization_target(polymer_MW, LA_GA, drug_polymer, surfactant_concentration, surfactant_HLB, aqueous_organic, pH, solvent_polarity_index):
                test_row = pd.DataFrame([[
                    polymer_MW, LA_GA, opt_m_mw, opt_m_logp, opt_m_tpsa, opt_m_mp,
                    opt_m_ha, opt_m_hd, opt_m_het, drug_polymer,
                    surfactant_concentration, surfactant_HLB, aqueous_organic, pH, solvent_polarity_index
                ]], columns=feature_cols)
                
                prediction = eval_model.predict(test_row)[0]
                return -abs(prediction - target_size)

            # Set the bounding boxes for variables that can be altered in the lab
            bounds = {
                'polymer_MW': (5000, 150000),
                'LA_GA': (0.0, 1.0),
                'drug_polymer': (0.01, 0.5),
                'surfactant_concentration': (0.1, 5.0),
                'surfactant_HLB': (1.0, 20.0),
                'aqueous_organic': (1.0, 20.0),
                'pH': (1.0, 14.0),
                'solvent_polarity_index': (0.0, 10.0)
            }

            optimizer = BayesianOptimization(f=optimization_target, pbounds=bounds, random_state=42)
            optimizer.maximize(init_points=10, n_iter=25)

            best_recipe = optimizer.max['params']
            max_error = optimizer.max['target']

            # Display Recommended Recipe Recipe
            st.success(f"🎯 Optimized Recipe Found! Estimated error: {abs(max_error):.2f} nm")
            
            recipe_df = pd.DataFrame({
                "Parameter Description": [
                    "Polymer Molecular Weight (Da)", "LA/GA Ratio", "Drug/Polymer Ratio",
                    "Surfactant Concentration (%)", "Surfactant HLB Value", "Aqueous/Organic Phase Ratio",
                    "Target Process pH", "Solvent Polarity Index"
                ],
                "Engine Value Output": [
                    f"{best_recipe['polymer_MW']:.0f}", f"{best_recipe['LA_GA']:.2f}", f"{best_recipe['drug_polymer']:.3f}",
                    f"{best_recipe['surfactant_concentration']:.2f}%", f"{best_recipe['surfactant_HLB']:.1f}", f"{best_recipe['aqueous_organic']:.2f}",
                    f"{best_recipe['pH']:.2f}", f"{best_recipe['solvent_polarity_index']:.2f}"
                ]
            })
            st.table(recipe_df)
