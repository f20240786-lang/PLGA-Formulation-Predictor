import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import matplotlib.pyplot as plt

# =====================================================================
# SYSTEM CONFIGURATION & INITIALIZATION
# =====================================================================

# Disable Optuna's noisy console outputs to keep clean production logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Page Layout Setup
st.set_page_config(
    page_title="PLGA Formulation & Design Hub", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧪 PLGA Nanoparticle AI Design Suite")

# Scientific Regulatory Disclaimer (Standard Practice for Material Discovery Softwares)
st.caption("🔬 *Disclaimer: Predictions are generated via machine learning architectures intended to support formulation design and should be validated experimentally.*")

# Exact training feature sequence expected by underlying model matrices
FEATURE_COLUMNS = [
    'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
    'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
    'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index'
]

# Historical Training Reference Benchmarks (Hardcoded to handle decoupled standalone setups)
METRICS_BENCHMARKS = pd.DataFrame({
    "Model Architecture": ["Stacking Ensemble", "Random Forest", "Gradient Boosting", "Extra Trees", "SVR", "XGBoost"],
    "Train R²": [0.962, 0.941, 0.925, 0.950, 0.812, 0.938],
    "Test R²": [0.895, 0.864, 0.851, 0.872, 0.764, 0.849],
    "RMSE (nm)": [14.2, 18.5, 20.1, 17.8, 28.3, 21.2]
})

# Training Distribution Reference Ranges for Real-Time Validation Guardrails
TRAINING_BOUNDS = {
    'polymer_MW': (2.4, 98.0), 
    'LA/GA': (1.0, 3.0), 
    'mol_MW': (244.27, 853.92),
    'mol_logP': (0.64, 4.33), 
    'mol_TPSA': (32.34, 224.45), 
    'mol_melting_point': (110.1, 267.5),
    'mol_Hacceptors': (1, 14), 
    'mol_Hdonors': (0, 5), 
    'mol_heteroatoms': (3, 15),
    'drug/polymer': (0.006, 1.0), 
    'surfactant_concentration': (0.02, 2.0),
    'surfactant_HLB': (13.0, 29.0), 
    'aqueous/organic': (1.78, 16.0), 
    'pH': (-2.0, 1.0),
    'solvent_polarity_index': (5.0, 5.5)
}

# =====================================================================
# CORE ENGINES & CORE FUNCTIONAL REQUISITES
# =====================================================================

@st.cache_resource
def load_all_models():
    """Securely unpickles and caches all machine learning models into RAM."""
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
    st.error(f"⚠️ Missing a required model file (.pkl)! Please check your root repository directory. Error: {e}")
    st.stop()

def validate_input_distribution(input_df):
    """Flags any parameter running out of boundary distributions via real-time statistical warnings."""
    out_of_bounds = False
    for col in FEATURE_COLUMNS:
        val = input_df[col].iloc[0]
        low, high = TRAINING_BOUNDS[col]
        if val < low or val > high:
            out_of_bounds = True
            break
    if out_of_bounds:
        st.warning("⚠️ One or more inputs are outside the range used during model training. Predictions may be less reliable due to statistical extrapolation.")

def calculate_confidence_tier(predictions):
    """Computes a consensus metric based on mathematical variance between model arrays."""
    std_dev = np.std(predictions)
    if std_dev < 15: 
        return "🔥 High Ensemble Agreement (Predictions converge perfectly)", "success"
    elif std_dev < 35: 
        return "⚠️ Moderate Ensemble Agreement (Expect minor variance in physical testing)", "warning"
    else: 
        return "🚨 Low Ensemble Agreement (High variance; formulate with extreme caution)", "error"

def draw_radar_chart(predictions_dict):
    """Builds a closed polar spider plot mapping core consensus alignment landscapes."""
    categories = list(predictions_dict.keys())
    values = list(predictions_dict.values())
    
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="teal", alpha=0.25)
    ax.plot(angles, values, color="teal", linewidth=2, marker='o')
    
    ax.set_yticklabels([]) 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, fontweight='bold')
    plt.tight_layout()
    return fig

def render_adaptive_shap(model, model_name, input_df):
    """Configures adaptive SHAP local attributions with explicit fallback for stacking/kernel models."""
    st.markdown("#### 🧠 SHAP Feature Contribution Analysis")
    try:
        # Route through highly optimized TreeExplainer arrays if handling tree structural ensembles
        if any(keyword in model_name for keyword in ["Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            if isinstance(shap_values, list): 
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                shap_values = shap_values[0]
        else:
            # Stacking/SVR models require a reference dataset. We construct a baseline reference centered on boundaries.
            baseline_row = {}
            for col in FEATURE_COLUMNS:
                low, high = TRAINING_BOUNDS[col]
                baseline_row[col] = (low + high) / 2.0
            baseline_df = pd.DataFrame([baseline_row], columns=FEATURE_COLUMNS)
            
            # Use safe KernelExplainer explicitly optimized for custom function models
            explainer = shap.KernelExplainer(model.predict, baseline_df)
            shap_values = explainer.shap_values(input_df, nsamples=100)
            
            # Flatten matrix dimension variants if returned inside extra wrapper lists
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        # Draw local contribution horizontal bar charts safely
        fig, ax = plt.subplots(figsize=(6, 3))
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:7] 
        features_to_plot = [FEATURE_COLUMNS[i] for i in sorted_idx]
        weights_to_plot = [shap_values[i] for i in sorted_idx]
        
        colors = ['#ff0051' if w >= 0 else '#008bfb' for w in weights_to_plot]
        ax.barh(features_to_plot[::-1], weights_to_plot[::-1], color=colors[::-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("🔴 Crimson (Right): Pushes size larger | 🔵 Sapphire (Left): Drives size smaller")
    except Exception as e:
        st.info("💡 SHAP visualization could not extract local gradients. Interpret parameter variations using the dynamic consensus tracking panel above.")

# =====================================================================
# SYSTEM LAYOUT & SEPARATION NAVIGATION ARCHITECTURES
# =====================================================================

tab1, tab2 = st.tabs(["🔮 Property Predictor", "🧬 High-Precision Formulation Designer"])

# =====================================================================
# TAB 1: PROPERTY PREDICTOR (FORWARD OPERATIONS)
# =====================================================================
with tab1:
    st.subheader("🤖 Model Configuration")
    selected_model_name = st.selectbox("Choose the ML Architecture:", options=list(all_models.keys()), key="predict_model")
    active_model = all_models[selected_model_name]

    st.write("---")
    st.subheader("Adjust Parameters to Test a Formulation Matrix")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        p_mw = st.slider("Polymer MW (kDa)", min_value=1.0, max_value=120.0, value=30.0, step=0.1)
        l_g = st.slider("LA/GA Ratio", min_value=0.5, max_value=4.0, value=1.0, step=0.1)
        m_mw = st.number_input("Drug Molecular Weight (g/mol)", min_value=100.0, max_value=1200.0, value=430.0)
        m_logp = st.slider("Drug LogP", min_value=-2.0, max_value=6.0, value=2.8, step=0.1)
        m_tpsa = st.number_input("Drug TPSA (Å²)", min_value=10.0, max_value=350.0, value=95.0)
        m_mp = st.number_input("Melting Point (°C)", min_value=50.0, max_value=400.0, value=152.0)
        m_ha = st.slider("H-Bond Acceptors", 0, 25, 5)
    with col_s2:
        m_hd = st.slider("H-Bond Donors", 0, 12, 2)
        m_het = st.slider("Heteroatoms Count", 0, 30, 7)
        d_p = st.slider("Drug/Polymer Ratio", min_value=0.001, max_value=2.0, value=0.15, step=0.005)
        s_c = st.slider("Surfactant Concentration (%)", min_value=0.01, max_value=5.0, value=0.8, step=0.01)
        s_hlb = st.slider("Surfactant HLB Value", min_value=5.0, max_value=40.0, value=21.5, step=0.5)
        a_o = st.slider("Aqueous/Organic Ratio", min_value=1.0, max_value=25.0, value=4.0, step=0.1)
        ph_val = st.slider("pH Level (Coded)", min_value=-5, max_value=5, value=0, step=1)
        s_pol = st.number_input("Solvent Polarity Index", min_value=3.0, max_value=7.0, value=5.1, step=0.1)

    input_df = pd.DataFrame([[p_mw, l_g, m_mw, m_logp, m_tpsa, m_mp, m_ha, m_hd, m_het, d_p, s_c, s_hlb, a_o, ph_val, s_pol]], columns=FEATURE_COLUMNS)
    
    # Run Boundary Checks
    validate_input_distribution(input_df)
    
    predicted_val = active_model.predict(input_df)[0]

    # Map outputs for Radar Chart Convergence
    radar_preds = {
        "Stacking": round(all_models["Stacking Ensemble"].predict(input_df)[0], 1),
        "RandomForest": round(all_models["Random Forest Regressor"].predict(input_df)[0], 1),
        "GradBoost": round(all_models["Gradient Boosting Regressor"].predict(input_df)[0], 1),
        "ExtraTrees": round(all_models["Extra Trees Regressor"].predict(input_df)[0], 1),
        "XGBoost": round(all_models["XGBoost Regressor"].predict(input_df)[0], 1)
    }
    
    conf_msg, conf_type = calculate_confidence_tier(list(radar_preds.values()))

    st.write("---")
    st.markdown("### 📊 Prediction Analytics Output")
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.metric(label="Target Model Prediction", value=f"{predicted_val:.2f} nm")
        if conf_type == "success": st.success(conf_msg)
        elif conf_type == "warning": st.warning(conf_msg)
        else: st.error(conf_msg)
        
        st.markdown("**Ensemble Values:**")
        for k, v in radar_preds.items():
            st.write(f"🔹 **{k}**: {v} nm")
            
    with col_p2:
        st.markdown("<p style='text-align: center; font-weight: bold; font-size:14px;'>Consensus Landscape</p>", unsafe_allow_html=True)
        st.pyplot(draw_radar_chart(radar_preds))

    st.write("---")
    render_adaptive_shap(active_model, selected_model_name, input_df)
    
    st.write("---")
    st.markdown("#### 📉 Reference System Performance Benchmarks")
    st.dataframe(METRICS_BENCHMARKS, use_container_width=True)

# =====================================================================
# TAB 2: INVERSE FORMULATION DESIGNER (BACKWARD OPTIMIZATION)
# =====================================================================
with tab2:
    st.subheader("🎯 Calculate Recipe from Target Criteria")
    st.write("Input your specific fixed drug parameters. The high-precision engine will iterate through the unconstrained feature boundaries to structure the optimized lab configuration.")

    target_size = st.number_input("Enter Target Particle Size (nm):", min_value=85.0, max_value=335.0, value=190.0)

    st.markdown("#### 🔒 Step 1: Input Your Fixed Molecule Properties")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fixed_m_mw = st.number_input("Drug Molecular Weight (g/mol)", min_value=100.0, max_value=1200.0, value=392.41, key="inv_mw")
        fixed_m_logp = st.number_input("Drug LogP value", min_value=-2.0, max_value=6.0, value=2.08, key="inv_logp")
        fixed_m_tpsa = st.number_input("Drug TPSA (Å²)", min_value=10.0, max_value=350.0, value=100.59, key="inv_tpsa")
        fixed_m_mp = st.number_input("Melting Point (°C)", min_value=5
