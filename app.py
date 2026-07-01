import streamlit as st
import pandas as pd
import numpy as np
import joblib
from bayes_opt import BayesianOptimization

# 1. Page Configuration
st.set_page_config(page_title="PLGA Formulation & Design Hub", layout="centered")
st.title("🧪 PLGA Nanoparticle AI Design Suite")

# 2. Load Assets and Dataset Securely
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
    try:
        df = pd.read_csv('NP_dataset.csv')
        df_subset = df.iloc[:137].copy()
    except Exception as e:
        st.error(f"⚠️ Could not load NP_dataset.csv from repository! Error: {e}")
        df_subset = None

    return models, df_subset

try:
    all_models, df_clean = load_all_assets()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing model file! {e}")
    st.stop()

# 3. Create Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Property Predictor", "🧬 Bayesian Formulation Designer"])

feature_cols = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
                'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index']

# =====================================================================
# TAB 1: PROPERTY PREDICTOR
# =====================================================================
with tab1:
    st.subheader("🤖 Model Configuration")
    selected_model_name = st.selectbox("Choose the ML Architecture:", options=list(all_models.keys()), key="predict_model")
    active_model = all_models[selected_model_name]

    st.write("---")
    st.subheader("Move Sliders to Test a Formulation")
    
    p_mw = st.slider("Polymer MW (kDa)", min_value=2.0, max_value=100.0, value=30.0, step=0.1)
    l_g = st.slider("LA/GA Ratio", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    m_mw = st.number_input("Drug Molecular Weight (g/mol)", min_value=240.0, max_value=860.0, value=430.0, key="p_m_mw")
    m_logp = st.slider("Drug LogP", min_value=0.5, max_value=4.5, value=2.8, step=0.1, key="p_m_logp")
    m_tpsa = st.number_input("Drug TPSA (Å²)", min_value=30.0, max_value=230.0, value=95.0, key="p_m_tpsa")
    m_mp = st.number_input("Melting Point (°C)", min_value=110.0, max_value=270.0, value=152.0, key="p_m_mp")
    m_ha = st.slider("H-Bond Acceptors", 1, 14, 5, key="p_m_ha")
    m_hd = st.slider("H-Bond Donors", 0, 5, 2, key="p_m_hd")
    m_het = st.slider("Heteroatoms Count", 3, 15, 7, key="p_m_het")
    d_p = st.slider("Drug/Polymer Ratio", min_value=0.005, max_value=1.0, value=0.15, step=0.005)
    s_c = st.slider("Surfactant Concentration (%)", min_value=0.02, max_value=2.0, value=0.8, step=0.01)
    s_hlb = st.slider("Surfactant HLB Value", min_value=13.0, max_value=29.0, value=21.5, step=0.5)
    a_o = st.slider("Aqueous/Organic Ratio", min_value=1.5, max_value=16.0, value=4.0, step=0.1)
    ph_val = st.slider("pH Level (Coded)", min_value=-2, max_value=1, value=0, step=1)
    s_pol = st.number_input("Solvent Polarity Index", min_value=5.0, max_value=5.5, value=5.1, step=0.1)

    input_df = pd.DataFrame([[p_mw, l_g, m_mw, m_logp, m_tpsa, m_mp, m_ha, m_hd, m_het, d_p, s_c, s_hlb, a_o, ph_val, s_pol]], columns=feature_cols)
    predicted_val = active_model.predict(input_df)[0]

    st.metric(label="Predicted Particle Size", value=f"{predicted_val:.2f} nm")

# =====================================================================
# TAB 2: HYBRID BAYESIAN DESIGNER (AUTO-LOOKUP + MANUAL CUSTOMIZATION)
# =====================================================================
with tab2:
    st.subheader("🎯 Inverse-Design a Brand New Formulation")
    st.write("Select an existing drug configuration or manually input completely new molecular metrics below.")

    opt_model_name = st.selectbox("Choose baseline model for optimization evaluation:", options=list(all_models.keys()), key="opt_model")
    target_size = st.number_input("Enter Target Particle Size (nm):", min_value=85.0, max_value=335.0, value=211.0)

    st.markdown("#### 💊 Step 1: Establish Your Drug Molecule Parameters")
    
    # 1. Create a hybrid option list
    dropdown_options = ["✨ Type Custom Molecule Properties Manually"]
    if df_clean is not None:
        # Add historical molecular weights to the menu list
        historical_mws = sorted(df_clean['mol_MW'].unique())
        dropdown_options.extend([f"Use Dataset Drug Signature (MW: {mw} g/mol)" for mw in historical_mws])

    selected_mode = st.selectbox("Select Drug Selection Mode:", options=dropdown_options)

    # 2. Determine base baseline defaults based on selection mode
    if selected_mode == "✨ Type Custom Molecule Properties Manually":
        # Global dataset averages work as safe initial text box defaults
        init_mw, init_logp, init_tpsa, init_mp, init_ha, init_hd, init_het, init_spol = 430.0, 2.8, 95.0, 152.0, 5, 2, 7, 5.1
        st.caption("📝 Inputs unlocked! Change the values below to evaluate an entirely unlisted drug compound.")
    else:
        # Extract the pure numeric MW string from the chosen option label
        parsed_mw = float(selected_mode.split("MW: ")[1].split(" g/mol")[0])
        drug_row = df_clean[df_clean['mol_MW'] == parsed_mw].iloc[0]
        
        init_mw = drug_row['mol_MW']
        init_logp = drug_row['mol_logP']
        init_tpsa = drug_row['mol_TPSA']
        init_mp = drug_row['mol_melting_point']
        init_ha = int(drug_row['mol_Hacceptors'])
        init_hd = int(drug_row['mol_Hdonors'])
        init_het = int(drug_row['mol_heteroatoms'])
        init_spol = drug_row['solvent_polarity_index']
        st.success(f"🔒 Row properties successfully fetched from your dataset file!")

    # 3. Create the numeric entry widgets (they automatically respond to the chosen mode values!)
    col1, col2 = st.columns(2)
    with col1:
        opt_m_mw = st.number_input("Drug Molecular Weight (g/mol)", value=init_mw, key="opt_val_mw")
        opt_m_logp = st.number_input("Drug LogP", value=init_logp, key="opt_val_logp")
        opt_m_tpsa = st.number_input("Drug TPSA (Å²)", value=init_tpsa, key="opt_val_tpsa")
        opt_m_mp = st.number_input("Melting Point (°C)", value=init_mp, key="opt_val_mp")
    with col2:
        opt_m_ha = st.number_input("H-Bond Acceptors", value=int(init_ha), key="opt_val_ha")
        opt_m_hd = st.number_input("H-Bond Donors", value=int(init_hd), key="opt_val_hd")
        opt_m_het = st.number_input("Heteroatoms Count", value=int(init_het), key="opt_val_het")
        opt_s_pol = st.number_input("Solvent Polarity Index", value=init_spol, key="opt_val_spol")

    # Optional Lab Constraint Feature
    st.markdown("#### 🔒 Step 2: Laboratory Resource Constraints")
    lock_polymer = st.checkbox("Force the engine to use a specific Polymer Molecular Weight (kDa)")
    if lock_polymer:
        fixed_poly_val = st.number_input("Specify Available Polymer MW (kDa):", min_value=2.4, max_value=98.0, value=45.0)

    # Run Button
    if st.button("🚀 Run Bayesian Formulation Generator"):
        with st.spinner("Bayesian engine evaluating structural multi-dimensional boundaries..."):
            eval_model = all_models[opt_model_name]

            if lock_polymer:
                def optimization_target(LA_GA, drug_polymer, surfactant_concentration, surfactant_HLB, aqueous_organic, pH):
                    test_row = pd.DataFrame([[
                        fixed_poly_val, LA_GA, opt_m_mw, opt_m_logp, opt_m_tpsa, opt_m_mp,
                        opt_m_ha, opt_m_hd, opt_m_het, drug_polymer,
                        surfactant_concentration, surfactant_HLB, aqueous_organic, pH, opt_s_pol
                    ]], columns=feature_cols)
                    prediction = eval_model.predict(test_row)[0]
                    return -abs(prediction - target_size)

                bounds = {
                    'LA_GA': (1.0, 3.0),
                    'drug_polymer': (0.006, 1.0),
                    'surfactant_concentration': (0.02, 2.0),
                    'surfactant_HLB': (13.0, 29.0),
                    'aqueous_organic': (1.78, 16.0),
                    'pH': (-2.0, 1.0)
                }
            else:
                def optimization_target(polymer_MW, LA_GA, drug_polymer, surfactant_concentration, surfactant_HLB, aqueous_organic, pH):
                    test_row = pd.DataFrame([[
                        polymer_MW, LA_GA, opt_m_mw, opt_m_logp, opt_m_tpsa, opt_m_mp,
                        opt_m_ha, opt_m_hd, opt_m_het, drug_polymer,
                        surfactant_concentration, surfactant_HLB, aqueous_organic, pH, opt_s_pol
                    ]], columns=feature_cols)
                    prediction = eval_model.predict(test_row)[0]
                    return -abs(prediction - target_size)

                bounds = {
                    'polymer_MW': (2.4, 98.0),
                    'LA_GA': (1.0, 3.0),
                    'drug_polymer': (0.006, 1.0),
                    'surfactant_concentration': (0.02, 2.0),
                    'surfactant_HLB': (13.0, 29.0),
                    'aqueous_organic': (1.78, 16.0),
                    'pH': (-2.0, 1.0)
                }

            optimizer = BayesianOptimization(f=optimization_target, pbounds=bounds, random_state=42)
            optimizer.maximize(init_points=15, n_iter=35)

            best_recipe = optimizer.max['params']
            max_error = optimizer.max['target']

            st.success(f"🎯 Optimized Recipe Found! Estimated error: {abs(max_error):.2f} nm")
            
            final_poly = fixed_poly_val if lock_polymer else best_recipe['polymer_MW']

            recipe_df = pd.DataFrame({
                "Parameter Description": [
                    "Polymer Molecular Weight (kDa)", "LA/GA Ratio", "Drug/Polymer Ratio",
                    "Surfactant Concentration (%)", "Surfactant HLB Value", "Aqueous/Organic Phase Ratio",
                    "Target Process pH (Coded)", "Solvent Polarity Index"
                ],
                "Engine Value Output": [
                    f"{final_poly:.1f}", f"{best_recipe['LA_GA']:.2f}", f"{best_recipe['drug_polymer']:.3f}",
                    f"{best_recipe['surfactant_concentration']:.2f}%", f"{best_recipe['surfactant_HLB']:.1f}", f"{best_recipe['aqueous_organic']:.2f}",
                    f"{best_recipe['pH']:.1f}", f"{opt_s_pol:.1f}"
                ]
            })
            st.table(recipe_df)
            
