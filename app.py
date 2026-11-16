"""Explain My Prediction - interactive explainable-ML demo (Streamlit)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.explain import Explainer, top_features
from src.model import MODELS, load_data, train

st.set_page_config(page_title="Explain My Prediction", page_icon="🔬", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.risk{border-radius:16px;padding:22px 26px;color:#fff}
.risk h1{margin:0;font-size:3rem;line-height:1}
.risk p{margin:6px 0 0;opacity:.9}
.low{background:linear-gradient(135deg,#0f766e,#14b8a6)}
.mid{background:linear-gradient(135deg,#b45309,#f59e0b)}
.high{background:linear-gradient(135deg,#b91c1c,#f43f5e)}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training model...")
def get_model(name):
    return train(name, load_data())


@st.cache_resource(show_spinner="Preparing explainer...")
def get_explainer(name):
    return Explainer(get_model(name))


@st.cache_data(show_spinner="Computing global explanations...")
def get_global(name):
    m, ex = get_model(name), get_explainer(name)
    return ex.explain(m.data.X_test)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔬 Explain My Prediction")
model_name = st.sidebar.selectbox("Model", list(MODELS), help="All three are trained on the same split.")
model, explainer = get_model(model_name), get_explainer(model_name)
X_test = model.data.X_test
global_exp = get_global(model_name)
features = top_features(global_exp, 8)

st.sidebar.markdown("---")
all_risk = model.malignant_probability(X_test)
default_idx = int(np.argmin(np.abs(all_risk - 0.5)))  # start with the most uncertain case: most interesting
idx = st.sidebar.selectbox(
    "Pick a real test sample", range(len(X_test)), index=default_idx, key=f"sample-{model_name}",
    format_func=lambda i: (f"Sample #{X_test.index[i]} · "
                           f"{'malignant' if model.data.y_test.iloc[i] == 0 else 'benign'} · model says {all_risk[i]:.0%}"))
st.sidebar.caption("Then change the most influential measurements below to see how the prediction reacts.")

row = X_test.iloc[[idx]].copy()
for f in features:
    lo, hi = float(model.data.X_train[f].min()), float(model.data.X_train[f].max())
    row[f] = st.sidebar.slider(f, lo, hi, float(row[f].iloc[0]), step=(hi - lo) / 200, key=f"{model_name}-{idx}-{f}")

p = float(model.malignant_probability(row)[0])
exp = explainer.explain(row)

# ------------------------------------------------------------------ header
st.title("Why did the model say that?")
st.caption("Breast-cancer tumour measurements (Wisconsin dataset) · explanations by SHAP · "
           "educational demo, **not medical advice**.")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "What-if", "Global insight", "Model quality"])

# ------------------------------------------------------------------ tab 1
with tab1:
    level = "low" if p < 0.3 else "mid" if p < 0.7 else "high"
    label = {"low": "Likely benign", "mid": "Uncertain", "high": "Likely malignant"}[level]
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown(f'<div class="risk {level}"><p>Predicted malignancy risk</p>'
                    f'<h1>{p:.0%}</h1><p>{label}</p></div>', unsafe_allow_html=True)
        st.write("")
        unit = {"probability": "probability points", "log-odds": "log-odds"}[explainer.space]
        st.caption(f"Bars on the right are in **{unit}**. Red pushes towards malignant, blue towards benign.")
        contrib = pd.DataFrame({"feature": exp.feature_names, "value": exp.data[0], "impact": exp.values[0]})
        contrib = contrib.reindex(contrib.impact.abs().sort_values(ascending=False).index).head(5)
        st.write("**Top 5 reasons**")
        for _, r in contrib.iterrows():
            tag = ":red[↑ malignant]" if r.impact > 0 else ":blue[↓ benign]"
            st.markdown(f"{tag} &nbsp; `{r.feature}` = {r.value:.3g}")
    with c2:
        fig = plt.figure(figsize=(7, 5))
        shap.plots.waterfall(exp[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Sweep one measurement while keeping the others fixed and watch the risk change "
             "(an *individual conditional expectation* curve).")
    feat = st.selectbox("Feature to vary", features)
    lo, hi = float(model.data.X_train[feat].min()), float(model.data.X_train[feat].max())
    grid = np.linspace(lo, hi, 60)
    sweep = pd.concat([row] * len(grid), ignore_index=True)
    sweep[feat] = grid
    risk = model.malignant_probability(sweep)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(grid, risk, color="#0d9488", lw=2.5)
    ax.axhline(0.5, color="#94a3b8", ls="--", lw=1)
    ax.scatter([row[feat].iloc[0]], [p], color="#e11d48", zorder=3, s=70, label="current value")
    ax.set_xlabel(feat); ax.set_ylabel("P(malignant)"); ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)
    flip = grid[np.argmax(risk >= 0.5)] if (risk >= 0.5).any() and (risk < 0.5).any() else None
    if flip is not None:
        st.info(f"The prediction crosses 50% around **{feat} ≈ {flip:.3g}** (others unchanged).")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.write("Which measurements matter most across the whole test set?")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Average impact (mean |SHAP|)**")
        plt.figure(figsize=(5, 5.5))
        shap.plots.bar(global_exp, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    with c2:
        st.write("**Impact vs. feature value (beeswarm)**")
        plt.figure(figsize=(5, 5.5))
        shap.plots.beeswarm(global_exp, max_display=12, show=False, plot_size=None)
        st.pyplot(plt.gcf(), clear_figure=True)

# ------------------------------------------------------------------ tab 4
with tab4:
    st.write(f"Held-out test set: **{len(X_test)}** samples.")
    cols = st.columns(len(model.metrics))
    for col, (k, v) in zip(cols, model.metrics.items()):
        col.metric(k, f"{v:.3f}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(model.confusion, cmap="GnBu")
        for (i, j), v in np.ndenumerate(model.confusion):
            ax.text(j, i, v, ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["benign", "malignant"]); ax.set_yticks([0, 1], ["benign", "malignant"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.write("**Why trust the explanations?**")
        err = explainer.max_error(X_test)
        st.write("SHAP values are *additive*: base value + sum of impacts equals the model output, "
                 "so the waterfall bars always add up to the prediction.")
        st.write(f"Largest reconstruction error on the test set: `{err:.1e}`")
        st.caption("Dataset: sklearn `load_breast_cancer` (569 samples, 30 features).")