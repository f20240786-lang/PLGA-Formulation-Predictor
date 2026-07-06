import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import matplotlib.pyplot as plt

# =====================================================================
# SYSTEM CONFIGURATION
# =====================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(
    page_title="PLGA Formulation & Design Hub",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧪 PLGA Nanoparticle AI Design Suite")

st.caption("Predictions are ML-based and require experimental validation.")

FEATURE_COLUMNS = [
    'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point',
    'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer',
    'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH',
    'solvent_polarity_index'
]

@st.cache_resource
def load_all_models():
    return {
        "Stacking Ensemble": joblib.load('stack_model.pkl'),
        "Random Forest Regressor": joblib.load('rf_model.pkl'),
        "Gradient Boosting Regressor": joblib.load('gbr_model.pkl'),
        "Extra Trees Regressor": joblib.load('etr_model.pkl'),
        "Support Vector Regressor": joblib.load('svr_model.pkl'),
        "XGBoost Regressor": joblib.load('xgb_model.pkl')
    }

all_models = load_all_models()

# =====================================================================
# UTILITIES
# =====================================================================

def calculate_confidence_tier(predictions):
    std_dev = np.std(predictions)
    if std_dev < 15:
        return "High Ensemble Agreement", "success"
    elif std_dev < 35:
        return "Moderate Ensemble Agreement", "warning"
    return "Low Ensemble Agreement", "error"


def draw_radar_chart(predictions_dict):
    keys = list(predictions_dict.keys())
    values = list(predictions_dict.values())

    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_yticklabels([])

    return fig


# =====================================================================
# SHAP (FIXED + CACHED)
# =====================================================================

@st.cache_resource
def get_explainer(model, model_name, background):
    if any(x in model_name for x in ["Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost"]):
        return shap.TreeExplainer(model)
    return shap.KernelExplainer(model.predict, background)


def render_shap(model, model_name, input_df):
    st.markdown("#### SHAP Feature Impact")

    try:
        background = input_df.copy()
        explainer = get_explainer(model, model_name, background)

        shap_values = explainer.shap_values(input_df, nsamples=30)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        idx = np.argsort(np.abs(shap_values))[::-1][:7]

        fig, ax = plt.subplots()
        ax.barh(
            [FEATURE_COLUMNS[i] for i in idx][::-1],
            shap_values[idx][::-1]
        )
        st.pyplot(fig)

    except Exception:
        st.info("SHAP could not be rendered for this model.")


# =====================================================================
# SESSION STATE CONTROL (FIX OPTUNA RERUN BUG)
# =====================================================================

if "optuna_running" not in st.session_state:
    st.session_state.optuna_running = False


# =====================================================================
# TABS
# =====================================================================

tab1, tab2 = st.tabs(["Predictor", "Inverse Design"])

# =====================================================================
# TAB 1
# =====================================================================

with tab1:
    model_name = st.selectbox("Model", list(all_models.keys()))
    model = all_models[model_name]

    inputs = {
        'polymer_MW': st.slider("Polymer MW", 1.0, 120.0, 30.0),
        'LA/GA': st.slider("LA/GA", 0.5, 4.0, 1.0),
        'mol_MW': st.number_input("MW", 100.0, 1200.0, 430.0),
        'mol_logP': st.slider("LogP", -2.0, 6.0, 2.8),
        'mol_TPSA': st.number_input("TPSA", 10.0, 350.0, 95.0),
        'mol_melting_point': st.number_input("MP", 50.0, 400.0, 152.0),
        'mol_Hacceptors': st.slider("H Acceptors", 0, 25, 5),
        'mol_Hdonors': st.slider("H Donors", 0, 12, 2),
        'mol_heteroatoms': st.slider("Heteroatoms", 0, 30, 7),
        'drug/polymer': st.slider("Drug/Polymer", 0.001, 2.0, 0.15),
        'surfactant_concentration': st.slider("Surfactant", 0.01, 5.0, 0.8),
        'surfactant_HLB': st.slider("HLB", 5.0, 40.0, 21.5),
        'aqueous/organic': st.slider("A/O", 1.0, 25.0, 4.0),
        'pH': st.slider("pH", -5, 5, 0),
        'solvent_polarity_index': st.number_input("SPI", 3.0, 7.0, 5.1)
    }

    input_df = pd.DataFrame([inputs])[FEATURE_COLUMNS]

    pred = model.predict(input_df)[0]

    ensemble_preds = {
        name: m.predict(input_df)[0]
        for name, m in all_models.items()
    }

    msg, level = calculate_confidence_tier(list(ensemble_preds.values()))

    st.metric("Prediction", round(pred, 2))
    st.write(msg)
    st.pyplot(draw_radar_chart(ensemble_preds))

    render_shap(model, model_name, input_df)


# =====================================================================
# TAB 2 (OPTIMIZED + FIXED)
# =====================================================================

with tab2:
    target_model_name = st.selectbox("Target Model", list(all_models.keys()))
    target_model = all_models[target_model_name]

    target_size = st.number_input("Target Size", 85.0, 335.0, 190.0)

    mw = st.number_input("MW", 100.0, 1200.0, 392.0)
    logp = st.number_input("LogP", -2.0, 6.0, 2.0)
    tpsa = st.number_input("TPSA", 10.0, 350.0, 100.0)
    mp = st.number_input("MP", 50.0, 400.0, 267.0)
    ha = st.number_input("H Acceptors", 0, 25, 6)
    hd = st.number_input("H Donors", 0, 12, 3)
    het = st.number_input("Heteroatoms", 0, 30, 9)
    spi = st.number_input("SPI", 3.0, 7.0, 5.1)

    if st.button("Run Optimization") and not st.session_state.optuna_running:

        st.session_state.optuna_running = True

        def objective(trial):

            params = {
                'polymer_MW': trial.suggest_float("polymer_MW", 2.4, 98.0),
                'LA/GA': trial.suggest_float("LA/GA", 1.0, 3.0),
                'drug/polymer': trial.suggest_float("drug/polymer", 0.006, 1.0),
                'surfactant_concentration': trial.suggest_float("surfactant_concentration", 0.02, 2.0),
                'surfactant_HLB': trial.suggest_float("surfactant_HLB", 13.0, 29.0),
                'aqueous/organic': trial.suggest_float("aqueous/organic", 1.78, 16.0),
                'pH': trial.suggest_float("pH", -5.0, 5.0)
            }

            row = {
                **params,
                'mol_MW': mw,
                'mol_logP': logp,
                'mol_TPSA': tpsa,
                'mol_melting_point': mp,
                'mol_Hacceptors': ha,
                'mol_Hdonors': hd,
                'mol_heteroatoms': het,
                'solvent_polarity_index': spi
            }

            df = pd.DataFrame([row])[FEATURE_COLUMNS]

            pred = target_model.predict(df)[0]

            return abs(pred - target_size)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200)

        st.session_state.optuna_running = False

        best = study.best_params

        st.success("Optimization complete")
        st.json(best)
