import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Maternity Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: linear_gradient(135deg, #D1FAE5, #A7F3D0);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2563EB;
    }
    .stButton>button {
        background: linear_gradient(135deg, #2563EB, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .sidebar-info {
        background: #EFF6FF;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load / Train Model ───────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('readmission_model.pkl'):
        with open('readmission_model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        # Train inline if pickle not present
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        n = 500
        age = np.random.randint(18, 46, n)
        delivery_type = np.random.choice([0, 1], n, p=[0.6, 0.4])
        location      = np.random.choice([0, 1], n, p=[0.55, 0.45])
        complications = np.random.choice([0, 1], n, p=[0.7, 0.3])
        los           = np.random.randint(2, 10, n)
        labor_dur     = np.round(np.random.uniform(2, 24, n), 1)
        logit = -2.5 + 0.8*complications + 0.5*delivery_type + 0.4*location + 0.1*(age-30) - 0.1*(los-5)
        prob  = 1 / (1 + np.exp(-logit))
        y     = (np.random.random(n) < prob).astype(int)

        X = pd.DataFrame({'Age': age, 'LaborDuration': labor_dur, 'LOS': los,
                          'DeliveryType_enc': delivery_type, 'Location_enc': location,
                          'Complications_enc': complications})
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        return {'model': rf, 'features': list(X.columns), 'threshold': 0.35}


artifacts = load_model()
model     = artifacts['model']
features  = artifacts['features']
threshold = artifacts.get('threshold', 0.35)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 About This Tool")
    st.markdown("""
    <div class="sidebar-info">
    This dashboard predicts <b>30-day hospital readmission risk</b> for maternity patients using a
    Random Forest model trained on 500 patient records.<br><br>
    <b>Features used:</b><br>
    • Age & Labor Duration<br>
    • Length of Stay<br>
    • Delivery Type<br>
    • Location (Urban/Rural)<br>
    • Complications<br><br>
    <b>Model metrics:</b><br>
    • Accuracy: ~77%<br>
    • AUC-ROC: ~0.65<br>
    • Risk threshold: 35%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚖️ Ethics Note")
st.info("This tool is designed to **assist** clinical decision-making, not replace it. "
            "Always combine model output with clinical judgment.")

    st.markdown("---")
    st.markdown("### 📊 Fairness Audit")
    st.markdown("""
    **Delivery type gap:** ≤ 9.9% ✅
    **Location gap:** ≤ 4.5% ✅
    *Last audit: February 2026*
    """)


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Maternity Readmission Risk Predictor</h1>
    <p style="opacity:0.85; font-size:1.1rem; margin:0">
        Enter patient details below to assess 30-day readmission risk
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────
st.markdown("### 📋 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.slider("Patient Age", min_value=18, max_value=45, value=28, step=1)
    location = st.selectbox("Location", ["Urban", "Rural"])

with col2:
    st.markdown("**Delivery Details**")
    delivery_type = st.selectbox("Delivery Type", ["Vaginal", "Cesarean"])
    complications = st.selectbox("Complications", ["No", "Yes"])

with col3:
    st.markdown("**Clinical Metrics**")
    labor_duration = st.slider("Labor Duration (hours)", min_value=1.0, max_value=24.0, value=10.0, step=0.5)
    los = st.slider("Length of Stay (days)", min_value=2, max_value=14, value=4, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
if st.button("🔍 Calculate Readmission Risk"):

    input_data = pd.DataFrame([[
        age,
        labor_duration,
        los,
        1 if delivery_type == 'Cesarean' else 0,
        1 if location == 'Rural' else 0,
        1 if complications == 'Yes' else 0
    ]], columns=features)

    prob     = float(model.predict_proba(input_data)[0, 1])
    is_high  = prob >= threshold
    risk_pct = prob * 100

    st.markdown("---")
    st.markdown("### 🎯 Risk Assessment")

    res_col, gauge_col, factors_col = st.columns([1.3, 1.5, 1.2])

    with res_col:
        if is_high:
            st.markdown(f"""
            <div class="risk-high">
                <h1 style="color:#DC2626; font-size:3rem; margin:0">⚠️</h1>
                <h2 style="color:#DC2626; margin:0.3rem 0">HIGH RISK</h2>
                <h3 style="color:#7f1d1d; margin:0.3rem 0">{risk_pct:.1f}% probability</h3>
                <p style="color:#991b1b; margin-top:0.5rem; font-size:0.9rem">
                    Recommend: Schedule follow-up within 7 days
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h1 style="color:#059669; font-size:3rem; margin:0">✅</h1>
                <h2 style="color:#059669; margin:0.3rem 0">LOW RISK</h2>
                <h3 style="color:#065f46; margin:0.3rem 0">{risk_pct:.1f}% probability</h3>
                <p style="color:#064e3b; margin-top:0.5rem; font-size:0.9rem">
                    Standard post-discharge protocol
                </p>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        # Gauge chart
        fig, ax = plt.subplots(figsize=(4.5, 3), subplot_kw={'aspect': 'equal'})
        theta_start, theta_end = np.pi, 0

        # Background arc
        theta = np.linspace(theta_start, theta_end, 200)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=18, color='#E5E7EB', solid_capstyle='round')

        # Color zones
        for zone_start, zone_end, color in [
            (np.pi, np.pi*2/3, '#10B981'),
            (np.pi*2/3, np.pi/3, '#FBBF24'),
            (np.pi/3, 0, '#EF4444'),
        ]:
            ztheta = np.linspace(zone_start, zone_end, 100)
            ax.plot(np.cos(ztheta), np.sin(ztheta), linewidth=18, color=color, alpha=0.3)

        # Value arc
        filled_end = np.pi - prob * np.pi
        vtheta = np.linspace(theta_start, filled_end, 200)
        arc_color = '#EF4444' if is_high else '#10B981'
        ax.plot(np.cos(vtheta), np.sin(vtheta), linewidth=18, color=arc_color, solid_capstyle='round')

        # Needle
        angle = np.pi - prob * np.pi
        ax.arrow(0, 0, 0.65*np.cos(angle), 0.65*np.sin(angle),
                 head_width=0.05, head_length=0.05, fc='#1F2937', ec='#1F2937', linewidth=2)
        ax.add_patch(plt.Circle((0, 0), 0.07, color='#1F2937', zorder=5))

        ax.text(0, -0.25, f"{risk_pct:.1f}%", ha='center', va='center',
                fontsize=22, fontweight='bold', color='#1F2937')
        ax.text(0, -0.45, "Readmission Probability", ha='center', fontsize=8.5, color='#6B7280')

        for label, x, y in [('Low', -0.85, -0.2), ('Medium', 0, 0.9), ('High', 0.85, -0.2)]:
            ax.text(x, y, label, ha='center', fontsize=8, color='#6B7280')

        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 1.1)
        ax.axis('off')
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with factors_col:
        st.markdown("**Risk Factor Summary**")

        factor_items = [
            ("Age", age, 30, "younger" , "older"),
            ("LOS", los, 5, "shorter", "longer"),
            ("Labor Dur.", labor_duration, 12, "shorter", "longer"),
        ]

        for name, val, ref, low_label, high_label in factor_items:
            direction = "↑ Higher risk" if val > ref else "↓ Lower risk"
            color = "#EF4444" if val > ref else "#10B981"
            st.markdown(f"**{name}:** {val} → <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)

        st.markdown("---")
        flag_style = lambda cond: "🔴" if cond else "🟢"
        st.markdown(f"{flag_style(delivery_type=='Cesarean')} **{delivery_type}** delivery")
        st.markdown(f"{flag_style(location=='Rural')} **{location}** location")
        st.markdown(f"{flag_style(complications=='Yes')} Complications: **{complications}**")

    # Clinical recommendations
    st.markdown("---")
    st.markdown("### 💊 Clinical Recommendations")
    if is_high:
        rcols = st.columns(3)
        with rcols[0]:
            st.error("**📞 Schedule Follow-up**\nBook appointment within 7 days of discharge")
        with rcols[1]:
            st.warning("**📋 Enhanced Discharge Plan**\nProvide written instructions for warning signs")
        with rcols[2]:
            st.info("**🩺 Home Nursing**\nConsider home nursing visit within 48–72 hours")
    else:
        rcols = st.columns(3)
        with rcols[0]:
            st.success("**📅 Standard Follow-up**\nRoutine 6-week postnatal appointment")
        with rcols[1]:
            st.success("**📞 Helpline**\nProvide helpline number for questions")
        with rcols[2]:
            st.success("**📱 Patient App**\nEnroll in standard maternity tracking app")

    st.caption(f"⚠️ This tool is for clinical decision support only. Risk threshold: {threshold:.0%}. "
               f"All decisions should be made by qualified medical professionals.")


# ── Batch Analysis Tab ───────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Population Analysis (Demo)")

demo_df = pd.DataFrame({
    'Age': [22, 35, 40, 28, 33, 41, 19, 38, 45, 25],
    'LaborDuration': [8, 20, 5, 15, 12, 3, 18, 22, 6, 10],
    'LOS': [3, 7, 4, 5, 6, 8, 3, 9, 5, 4],
    'DeliveryType': ['Vaginal','Cesarean','Vaginal','Cesarean','Vaginal','Cesarean','Vaginal','Cesarean','Vaginal','Vaginal'],
    'Location': ['Urban','Rural','Urban','Rural','Urban','Rural','Urban','Urban','Rural','Urban'],
    'Complications': ['No','Yes','No','Yes','No','Yes','No','Yes','No','No'],
})

for idx, row in demo_df.iterrows():
    inp = pd.DataFrame([[
        row['Age'], row['LaborDuration'], row['LOS'],
        1 if row['DeliveryType']=='Cesarean' else 0,
        1 if row['Location']=='Rural' else 0,
        1 if row['Complications']=='Yes' else 0
    ]], columns=features)
    demo_df.loc[idx, 'Risk_Prob'] = float(model.predict_proba(inp)[0,1])
    demo_df.loc[idx, 'Risk_Level'] = 'HIGH' if demo_df.loc[idx, 'Risk_Prob'] >= threshold else 'LOW'

demo_df['Risk_Prob'] = (demo_df['Risk_Prob']*100).round(1).astype(str) + '%'

def color_risk(val):
    if val == 'HIGH':
        return 'background-color: #FEE2E2; color: #DC2626; font-weight: bold'
    return 'background-color: #D1FAE5; color: #059669; font-weight: bold'

styled = demo_df.style.applymap(color_risk, subset=['Risk_Level'])
st.dataframe(styled, use_container_width=True)

st.markdown("""
<p style='text-align:center; color:#6B7280; font-size:0.85rem; margin-top:1rem'>
🏥 Maternity Readmission Risk Predictor | Built with Streamlit & scikit-learn | For clinical decision support only
</p>
""", unsafe_allow_html=True)
