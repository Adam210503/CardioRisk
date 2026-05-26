import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk · Clinical Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS — clinical, refined, dark-accented
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2a38;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #8b949e !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    background: #1f6feb !important;
    color: #fff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Sidebar section divider label */
.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1f6feb;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2a38;
}

/* ── Main area ── */
.main-header {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.header-badge {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 28px;
    line-height: 1;
}
.header-title {
    font-size: 24px;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.header-subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}
.header-disclaimer {
    margin-left: auto;
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    color: #92400e;
    font-weight: 500;
    max-width: 240px;
    text-align: right;
    line-height: 1.5;
}

/* ── Patient summary card ── */
.patient-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}
.patient-card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 12px;
}
.patient-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}
.patient-metric {
    text-align: center;
}
.patient-metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    color: #0f172a;
    display: block;
}
.patient-metric-label {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-top: 2px;
    display: block;
}

/* ── Analyse button ── */
div[data-testid="stButton"] > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1e40af !important;
}

/* ── Risk result panels ── */
.result-wrapper {
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    border: 1px solid transparent;
}
.result-low    { background: #f0fdf4; border-color: #86efac; }
.result-medium { background: #fffbeb; border-color: #fcd34d; }
.result-high   { background: #fff1f2; border-color: #fda4af; }

.result-tier {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.tier-low    { color: #15803d; }
.tier-medium { color: #b45309; }
.tier-high   { color: #be123c; }

.result-headline {
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.result-body {
    font-size: 13px;
    color: #475569;
    line-height: 1.6;
}

/* ── Probability table ── */
.prob-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
}
.prob-table th {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #e2e8f0;
}
.prob-table td {
    padding: 8px 10px;
    color: #334155;
    border-bottom: 1px solid #f1f5f9;
    font-family: 'DM Mono', monospace;
}
.prob-table tr:last-child td { border-bottom: none; }
.prob-bar-bg {
    background: #f1f5f9;
    border-radius: 99px;
    height: 6px;
    width: 100%;
}
.prob-bar-fill {
    background: #1d4ed8;
    border-radius: 99px;
    height: 6px;
}
.prob-bar-fill.active { background: #be123c; }

/* ── Feature flags ── */
.flag-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}
.flag-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #475569;
}
.flag-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.flag-dot.risk   { background: #ef4444; }
.flag-dot.normal { background: #22c55e; }

/* ── Section label ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Expander style */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('optimized_heart_disease_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python3 train_model.py` first.")
    st.stop()


# ─────────────────────────────────────────────
# Sidebar — clinical inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:15px; font-weight:600; color:#e6edf3; letter-spacing:-0.01em;'>CardioRisk</div>
        <div style='font-size:11px; color:#8b949e; margin-top:2px;'>Clinical Assessment System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Demographics</div>", unsafe_allow_html=True)
    age  = st.slider("Age (years)", 20, 100, 54)
    sex  = st.selectbox("Biological Sex", [0, 1],
                        format_func=lambda x: "Female" if x == 0 else "Male")

    st.markdown("<div class='sidebar-section'>Resting Vitals</div>", unsafe_allow_html=True)
    cp       = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                             format_func=lambda x: {1: "Typical Angina (classic exertional)", 
                                                    2: "Atypical Angina (partial presentation)",
                                                    3: "Non-anginal Chest Pain (e.g. musculoskeletal)",
                                                    4: "No Chest Pain / Asymptomatic"
                                                    }[x])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 246)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                             format_func=lambda x: {0:"Normal", 1:"ST-T Wave Abnormality",
                                                    2:"Left Ventricular Hypertrophy"}[x])

    st.markdown("<div class='sidebar-section'>Stress Test</div>", unsafe_allow_html=True)
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
    exang   = st.selectbox("Exercise-Induced Angina", [0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope   = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3],
                            format_func=lambda x: {1:"Upsloping", 2:"Flat", 3:"Downsloping"}[x])

    st.markdown("<div class='sidebar-section'>Structural Indicators</div>", unsafe_allow_html=True)
    ca   = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [3, 6, 7],
                         format_func=lambda x: {3: "Normal — no perfusion defect", 6: "Fixed Defect — scarred tissue (prior infarct)",7: "Reversible Defect — ischaemia under stress"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("Analyse Cardiac Risk Profile", type="primary")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='header-badge'>🫀</div>
    <div>
        <p class='header-title'>Cardiac Risk Assessment</p>
        <p class='header-subtitle'>Multi-class predictive model · Cleveland Heart Disease dataset · FastAPI + Streamlit</p>
    </div>
    <div class='header-disclaimer'>
        ⚠ For clinical decision support only.<br>Not a substitute for physician judgement.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Patient snapshot (always visible)
# ─────────────────────────────────────────────
cp_labels  = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "No Chest Pain"}
ecg_labels = {0:"Normal", 1:"ST-T Abnorm.", 2:"LV Hypertrophy"}

st.markdown(f"""
<div class='patient-card'>
    <div class='patient-card-title'>Current Patient Profile</div>
    <div class='patient-grid'>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{age}</span>
            <span class='patient-metric-label'>Age</span>
        </div>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{'M' if sex == 1 else 'F'}</span>
            <span class='patient-metric-label'>Sex</span>
        </div>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{trestbps}</span>
            <span class='patient-metric-label'>BP (mmHg)</span>
        </div>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{chol}</span>
            <span class='patient-metric-label'>Chol (mg/dL)</span>
        </div>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{thalach}</span>
            <span class='patient-metric-label'>Max HR</span>
        </div>
        <div class='patient-metric'>
            <span class='patient-metric-value'>{oldpeak}</span>
            <span class='patient-metric-label'>ST Depress.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Inference + results
# ─────────────────────────────────────────────
BACKEND_URL = "http://backend:8000/predict"

input_data = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
    columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
)

if analyse:
    # Package the raw inputs into a clean JSON payload for the network call
    payload = {
        "age": float(age), "sex": float(sex), "cp": float(cp), "trestbps": float(trestbps),
        "chol": float(chol), "fbs": float(fbs), "restecg": float(restecg), "thalach": float(thalach),
        "exang": float(exang), "oldpeak": float(oldpeak), "slope": float(slope), "ca": float(ca), "thal": float(thal)
    }
    
    try:
        # Route the request to the containerized FastAPI service instead of evaluating locally
        response = requests.post(BACKEND_URL, json=payload)
        result = response.json()
        
        # Populate variables from the backend API response layer
        prediction = result["severity_class"]
        probabilities = np.array(result["probabilities"])
        max_prob = probabilities[prediction]
        
    except requests.exceptions.ConnectionError:
        st.error("🚨 Container Network Disconnect! Verify that the backend service container is healthy and responding.")
        st.stop()

    col_result, col_detail = st.columns([1, 1], gap="large")

    # ── Left: risk verdict ──
    with col_result:
        if prediction == 0:
            css_cls   = "result-low"
            tier_css  = "tier-low"
            tier_txt  = "Low Risk"
            headline  = "No significant cardiac disease detected"
            body      = ("The patient's current clinical profile does not indicate structural "
                         "heart disease. Routine monitoring and preventive care are advised. "
                         "Re-assess if symptoms develop.")
        elif prediction in [1, 2]:
            css_cls   = "result-medium"
            tier_css  = "tier-medium"
            tier_txt  = f"Medium Risk — Severity Class {prediction}"
            headline  = "Mild to moderate coronary narrowing possible"
            body      = ("The model identifies features consistent with early-to-moderate "
                         "arterial disease. Further non-invasive testing (stress echo, CT angiography) "
                         "is clinically recommended before any intervention.")
        else:
            css_cls   = "result-high"
            tier_css  = "tier-high"
            tier_txt  = f"High Risk — Severity Class {prediction}"
            headline  = "Significant multi-vessel disease highly probable"
            body      = ("The model flags a high-severity profile. The patient's haemodynamic "
                         "and structural indicators are consistent with critical coronary artery disease. "
                         "Urgent cardiology referral and invasive assessment are strongly indicated.")

        st.markdown(f"""
        <div class='result-wrapper {css_cls}'>
            <div class='result-tier {tier_css}'>{tier_txt}</div>
            <div class='result-headline'>{headline}</div>
            <div class='result-body'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.markdown(f"""
        <div style='font-size:12px; color:#64748b; margin-top:-8px; margin-bottom:16px;'>
            Model confidence: <strong style='color:#0f172a; font-family:DM Mono,monospace;'>{max_prob:.1%}</strong>
            &nbsp;·&nbsp; Predicted severity class: <strong style='color:#0f172a; font-family:DM Mono,monospace;'>{prediction}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Clinical flags
        st.markdown("<div class='section-label'>Clinical Flags</div>", unsafe_allow_html=True)
        flags = [
            ("Chest pain type", cp_labels[cp], cp in [3, 4]),
            ("Hypertension",    f"{trestbps} mmHg", trestbps >= 140),
            ("Hypercholesterolaemia", f"{chol} mg/dL", chol >= 240),
            ("Fasting glucose", "Elevated" if fbs else "Normal", fbs == 1),
            ("Exercise angina", "Present" if exang else "Absent", exang == 1),
            ("ST depression",   f"{oldpeak}", oldpeak >= 2.0),
        ]
        flag_html = "<div class='flag-grid'>"
        for label, val, is_risk in flags:
            dot_cls = "risk" if is_risk else "normal"
            flag_html += f"""
            <div class='flag-item'>
                <div class='flag-dot {dot_cls}'></div>
                <div>
                    <div style='font-weight:500; color:#334155;'>{label}</div>
                    <div style='color:#94a3b8; font-size:11px;'>{val}</div>
                </div>
            </div>"""
        flag_html += "</div>"
        st.markdown(flag_html, unsafe_allow_html=True)

    # ── Right: probability breakdown ──
    with col_detail:
        st.markdown("<div class='section-label'>Class Probability Distribution</div>", unsafe_allow_html=True)

        severity_labels = {
            0: "No disease",
            1: "Mild narrowing",
            2: "Moderate narrowing",
            3: "Severe disease",
            4: "Critical disease",
        }

        rows = ""
        for i, prob in enumerate(probabilities):
            bar_pct   = prob * 100
            active    = "active" if i == prediction else ""
            rows += f"""
            <tr>
                <td style='color:#64748b; font-family: DM Sans, sans-serif; font-size:12px;'>
                    Class {i}<br>
                    <span style='color:#94a3b8; font-size:10px;'>{severity_labels[i]}</span>
                </td>
                <td>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill {active}' style='width:{bar_pct:.1f}%'></div>
                    </div>
                </td>
                <td style='text-align:right; color:{"#be123c" if i==prediction else "#334155"};
                           font-weight:{"600" if i==prediction else "400"};'>
                    {prob:.1%}
                </td>
            </tr>"""

        st.markdown(f"""
        <table class='prob-table'>
            <thead><tr>
                <th>Severity</th><th style='width:55%'>Probability</th><th>Score</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # Structural indicators summary
        st.markdown("<br><div class='section-label'>Structural Indicators</div>", unsafe_allow_html=True)
        thal_labels = {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}
        slope_labels = {1:"Upsloping", 2:"Flat", 3:"Downsloping"}
        struct_flags = [
            ("Vessels (fluoroscopy)", f"{ca} vessel{'s' if ca != 1 else ''}", ca >= 2),
            ("Thalassemia",           thal_labels[thal], thal == 7),
            ("ST slope",              slope_labels[slope], slope in [2, 3]),
            ("Resting ECG",           ecg_labels[restecg], restecg != 0),
        ]
        sf_html = "<div class='flag-grid'>"
        for label, val, is_risk in struct_flags:
            dot_cls = "risk" if is_risk else "normal"
            sf_html += f"""
            <div class='flag-item'>
                <div class='flag-dot {dot_cls}'></div>
                <div>
                    <div style='font-weight:500; color:#334155;'>{label}</div>
                    <div style='color:#94a3b8; font-size:11px;'>{val}</div>
                </div>
            </div>"""
        sf_html += "</div>"
        st.markdown(sf_html, unsafe_allow_html=True)

        # Raw data expander
        with st.expander("View raw input vector"):
            st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

else:
    # Pre-analysis placeholder
    st.markdown("""
    <div style='
        text-align:center;
        padding: 60px 20px;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        color: #94a3b8;
    '>
        <div style='font-size:36px; margin-bottom:12px;'>🫀</div>
        <div style='font-size:15px; font-weight:500; color:#64748b; margin-bottom:6px;'>
            Awaiting analysis
        </div>
        <div style='font-size:13px;'>
            Configure the patient's clinical metrics in the sidebar,<br>
            then click <strong style='color:#1d4ed8;'>Analyse Cardiac Risk Profile</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Inference via FastAPI Backend Connection
# ─────────────────────────────────────────────
BACKEND_URL = "http://backend:8000/predict"