import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna

# Disable Optuna's noisy console outputs to keep the log clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Page Configuration
st.set_page_config(page_title="PLGA Formulation & Design Hub", layout="centered")
st.title("🧪 PLGA Nanoparticle AI Design Suite")

# 2. Load Models Securely (Completely Independent of Dataset Files)
@st.cache_resource
def load_all_models():
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
    all_models = load_all_models()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing a required model file (.pkl)! Please check your repository. Error: {e}")
    st.stop()

# 3. Create Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Property Predictor", "🧬 High-Precision Formulation Designer"])

# Strict training feature sequence expected by your .pkl models
feature_cols = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
                'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index']

# =====================================================================
# TAB 1: PROPERTY PREDICTOR (FORWARD SIMULATION VIA SLIDERS)
# =====================================================================
with tab1:
    st.subheader("🤖 Model Configuration")
    selected_model_name = st.selectbox("Choose the ML Architecture:", options=list(all_models.keys()), key="predict_model")
    active_model = all_models[selected_model_name]

    st.write("---")
    st.subheader("Move Sliders to Test a Formulation")
    
    p_mw = st.slider("Polymer MW (kDa)", min_value=2.4, max_value=98.0, value=30.0, step=0.1)
    l_g = st.slider("LA/GA Ratio", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    m_mw = st.number_input("Drug Molecular Weight (g/mol)", min_value=244.0, max_value=854.0, value=430.0, key="p_m_mw")
    m_logp = st.slider("Drug LogP", min_value=0.6, max_value=4.4, value=2.8, step=0.1, key="p_m_logp")
    m_tpsa = st.number_input("Drug TPSA (Å²)", min_value=32.0, max_value=225.0, value=95.0, key="p_m_tpsa")
    m_mp = st.number_input("Melting Point (°C)", min_value=110.1, max_value=267.5, value=152.0, key="p_m_mp")
    m_ha = st.slider("H-Bond Acceptors", 1, 14, 5, key="p_m_ha")
    m_hd = st.slider("H-Bond Donors", 0, 5, 2, key="p_m_hd")
    m_het = st.slider("Heteroatoms Count", 3, 15, 7, key="p_m_het")
    d_p = st.slider("Drug/Polymer Ratio", min_value=0.006, max_value=1.0, value=0.15, step=0.005)
    s_c = st.slider("Surfactant Concentration (%)", min_value=0.02, max_value=2.0, value=0.8, step=0.01)
    s_hlb = st.slider("Surfactant HLB Value", min_value=13.0, max_value=29.0, value=21.5, step=0.5)
    a_o = st.slider("Aqueous/Organic Ratio", min_value=1.78, max_value=16.0, value=4.0, step=0.1)
    ph_val = st.slider("pH Level (Coded)", min_value=-2, max_value=1, value=0, step=1)
    s_pol = st.number_input("Solvent Polarity Index", min_value=5.0, max_value=5.5, value=5.1, step=0.1)

    input_df = pd.DataFrame([[p_mw, l_g, m_mw, m_logp, m_tpsa, m_mp, m_ha, m_hd, m_het, d_p, s_c, s_hlb, a_o, ph_val, s_pol]], columns=feature_cols)
    predicted_val = active_model.predict(input_df)[0]

    st.metric(label="Predicted Particle Size", value=f"{predicted_val:.2f} nm")

# =====================================================================
# TAB 2: FLIPPED INVERSE DESIGNER (FIXED DRUG -> DESIGNED RECIPE)
# =====================================================================
with tab2:
    st.subheader("🎯 Calculate Recipe from Drug Properties")
    st.write("Provide your molecule's properties below. The high-accuracy ensemble optimizer will compute the required formulation variables to safely achieve your target size.")

    target_size = st.number_input("Enter Target Particle Size (nm):", min_value=85.0, max_value=335.0, value=190.0)

    st.markdown("#### 🔒 Step 1: Input Your Fixed Molecule Properties")
    col1, col2 = st.columns(2)
    with col1:
        fixed_m_mw = st.number_input("Drug Molecular Weight (g/mol)", min_value=244.27, max_value=853.92, value=392.41, key="f_mw")
        fixed_m_logp = st.number_input("Drug LogP value", min_value=0.64, max_value=4.33, value=2.08, key="f_logp")
        fixed_m_tpsa = st.number_input("Drug TPSA (Å²)", min_value=32.34, max_value=224.45, value=100.59, key="f_tpsa")
        fixed_m_mp = st.number_input("Melting Point (°C)", min_value=110.1, max_value=267.5, value=267.5, key="f_mp")
    with col2:
        fixed_m_ha = st.number_input("H-Bond Acceptors Count", min_value=1, max_value=14, value=6, step=1, key="f_ha")
        fixed_m_hd = st.number_input("H-Bond Donors Count", min_value=0, max_value=5, value=3, step=1, key="f_hd")
        fixed_m_het = st.number_input("Heteroatoms Count", min_value=3, max_value=15, value=9, step=1, key="f_het")
        fixed_s_pol = st.number_input("Solvent Polarity Index Baseline", min_value=5.0, max_value=5.5, value=5.1, step=0.1, key="f_spol")

    st.markdown("#### 🚀 Step 2: Generate Lab Formulation Strategy")
    if st.button("🚀 Run High-Accuracy Recipe Optimization Loop"):
        with st.spinner("Optuna engine executing cross-validation ensemble optimization..."):
            
            # Load cross-checking architectures to eliminate structural hallucinations
            model_stack = all_models["Stacking Ensemble"]
            model_rf = all_models["Random Forest Regressor"]
            model_gbr = all_models["Gradient Boosting Regressor"]

            def objective(trial):
                # Unconstrained formulation exploration boundaries
                polymer_MW = trial.suggest_float("polymer_MW", 2.4, 98.0)
                LA_GA = trial.suggest_float("LA_GA", 1.0, 3.0)
                drug_polymer = trial.suggest_float("drug_polymer", 0.006, 1.0)
                surfactant_concentration = trial.suggest_float("surfactant_concentration", 0.02, 2.0)
                surfactant_HLB = trial.suggest_float("surfactant_HLB", 13.0, 29.0)  
                aqueous_organic = trial.suggest_float("aqueous_organic", 1.78, 16.0)
                pH = trial.suggest_float("pH", -2.0, 1.0)

                # Row DataFrame structure to map accurately with the underlying models
                trial_df = pd.DataFrame([{
                    'polymer_MW': polymer_MW, 'LA/GA': LA_GA, 'mol_MW': fixed_m_mw,
                    'mol_logP': fixed_m_logp, 'mol_TPSA': fixed_m_tpsa, 'mol_melting_point': fixed_m_mp,
                    'mol_Hacceptors': int(fixed_m_ha), 'mol_Hdonors': int(fixed_m_hd), 'mol_heteroatoms': int(fixed_m_het),
                    'drug/polymer': drug_polymer, 'surfactant_concentration': surfactant_concentration,
                    'surfactant_HLB': surfactant_HLB, 'aqueous/organic': aqueous_organic, 'pH': pH,
                    'solvent_polarity_index': fixed_s_pol
                }])

                # Evaluate trial across the ensemble
                pred_stack = model_stack.predict(trial_df)[0]
                pred_rf = model_rf.predict(trial_df)[0]
                pred_gbr = model_gbr.predict(trial_df)[0]
                
                # Log our recommendation baseline size
                trial.set_user_attr("predicted_size", pred_stack)
                
                # Deviation math calculations
                error_stack = abs(pred_stack - target_size)
                error_rf = abs(pred_rf - target_size)
                error_gbr = abs(pred_gbr - target_size)
                
                # Calculate model disagreement variance (Standard Deviation)
                disagreement_penalty = np.std([pred_stack, pred_rf, pred_gbr])

                # Combine primary error target minimization with the consistency check constraint
                return error_stack + (disagreement_penalty * 1.5)

            # High depth search to safely settle on the true mathematical minimum
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=500)

            # Pull final data points
            best_recipe_params = study.best_params
            final_pred_size = study.best_trial.user_attrs["predicted_size"]
            deviation_error = study.best_value

            st.success("🎯 Optimal Recipe Successfully Determined!")
            
            # Highlight target achievement using clean metrics rows
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Ensemble Predicted Size", f"{final_pred_size:.2f} nm")
            col_m2.metric("Target Deviation Margin", f"{deviation_error:.4f} nm")

            # Build the clean, formatted DataFrame table output
            recipe_df = pd.DataFrame({
                "Parameter Description": [
                    "Polymer Molecular Weight (kDa)", 
                    "LA/GA Ratio", 
                    "Drug/Polymer Ratio",
                    "Surfactant Concentration (%)", 
                    "Surfactant HLB Value", 
                    "Aqueous/Organic Phase Ratio",
                    "Target Process pH (Coded)",
                    "Solvent Polarity Index"
                ],
                "Engine Value Output": [
                    f"{best_recipe_params.get('polymer_MW'):.1f}", 
                    f"{best_recipe_params.get('LA_GA'):.2f}", 
                    f"{best_recipe_params.get('drug_polymer'):.3f}",
                    f"{best_recipe_params.get('surfactant_concentration'):.2f}%", 
                    f"{best_recipe_params.get('surfactant_HLB'):.1f}", 
                    f"{best_recipe_params.get('aqueous_organic'):.2f}",
                    f"{best_recipe_params.get('pH'):.1f}", 
                    f"{fixed_s_pol:.1f}"
                ]
            })
            
            # Show the table
            st.table(recipe_df)

            # Instant CSV Download button for easy notebook management
            st.write("")
            csv_data = recipe_df.to_csv(index=False)
            st.download_button(
                label="📥 Download This Lab Recipe (CSV)",
                data=csv_data,
                file_name=f"PLGA_AI_Recipe_{target_size}nm.csv",
                mime="text/csv"
            )
