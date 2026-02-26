%%writefile streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Maternity Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
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
        background: linear-gradient(135deg, #2563EB, #1d4ed8);
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


@st.cache_resource
def load_all_models_and_metrics():
    try:
        df = pd.read_csv('maternity_data.csv')
    except FileNotFoundError:
        st.info("📊 Using demo data (maternity_data.csv not found). For full functionality, upload `maternity_data.csv`.")
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'PatientID': range(1001, 1001 + n_samples),
            'Age': np.random.uniform(18, 45, n_samples),
            'DeliveryType': np.random.choice(['Vaginal', 'Cesarean'], n_samples),
            'LaborDuration': np.random.uniform(1, 16, n_samples),
            'Location': np.random.choice(['Urban', 'Rural'], n_samples),
            'Complications': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3]),
            'Readmitted': np.random.choice(['No', 'Yes'], n_samples, p=[0.75, 0.25]),
            'LengthofStaydays': np.random.uniform(2, 15, n_samples)
        })

    df = df[(df['Age'] >= 18) & (df['Age'] <= 45) & (df['LengthofStaydays'] >= 2)].copy()
    df['LaborDuration'] = df['LaborDuration'].fillna(df['LaborDuration'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Complications'] = df['Complications'].fillna(df['Complications'].mode()[0])

    df['Readmitted'] = (df['Readmitted'] == 'Yes').astype(int)
    df['Location_Encoded'] = (df['Location'] == 'Rural').astype(int)
    df['Complications_Encoded'] = (df['Complications'] == 'Yes').astype(int)

    df['Complication_Risk'] = df['Complications_Encoded'] * (df['LengthofStaydays'] / 5)
    df['LOS_Severity'] = df['LengthofStaydays'] ** 1.5
    df['Age_LOS_Interaction'] = (df['Age'] / 30) * df['LengthofStaydays']

    feature_cols = [
        'Age', 'LaborDuration', 'LengthofStaydays',
        'Location_Encoded', 'Complications_Encoded',
        'Complication_Risk', 'LOS_Severity', 'Age_LOS_Interaction'
    ]

    X = df[feature_cols]
    y = df['Readmitted']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_weight = np.ones(len(X_train))
    df_train = df.iloc[X_train.index]
    sample_weight[df_train['Complications_Encoded'].values == 1] = 2.0
    sample_weight[df_train['LengthofStaydays'].values > 10] = 1.5
    both_mask = (df_train['Complications_Encoded'].values == 1) & (df_train['LengthofStaydays'].values > 10)
    sample_weight[both_mask] = 3.0

    model_1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model_1.fit(X_train, y_train, sample_weight=sample_weight)

    model_2 = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2, min_samples_split=4,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_2.fit(X_train, y_train, sample_weight=sample_weight)

    model_3 = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8,
        min_samples_leaf=2, random_state=42
    )
    model_3.fit(X_train, y_train)

    model_4 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model_4.fit(X_train, y_train, sample_weight=sample_weight)

    try:
        import xgboost as xgb
        model_5 = xgb.XGBClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model_5.fit(X_train, y_train, sample_weight=sample_weight)
    except ImportError:
        model_5 = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
        model_5.fit(X_train, y_train, sample_weight=sample_weight)

    def evaluate_model_performance(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba_full = model.predict_proba(X_test)
        y_pred_proba = y_pred_proba_full[:, 1] if y_pred_proba_full.shape[1] > 1 else y_pred_proba_full[:, 0]
        return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred_proba)

    accuracy_1, auc_1 = evaluate_model_performance(model_1, X_test, y_test)
    accuracy_2, auc_2 = evaluate_model_performance(model_2, X_test, y_test)
    accuracy_3, auc_3 = evaluate_model_performance(model_3, X_test, y_test)
    accuracy_4, auc_4 = evaluate_model_performance(model_4, X_test, y_test)
    accuracy_5, auc_5 = evaluate_model_performance(model_5, X_test, y_test)

    return {
        'model_1': model_1, 'model_2': model_2, 'model_3': model_3,
        'model_4': model_4, 'model_5': model_5,
        'accuracy_1': accuracy_1, 'accuracy_2': accuracy_2, 'accuracy_3': accuracy_3,
        'accuracy_4': accuracy_4, 'accuracy_5': accuracy_5,
        'auc_1': auc_1, 'auc_2': auc_2, 'auc_3': auc_3,
        'auc_4': auc_4, 'auc_5': auc_5,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test
    }


all_artifacts = load_all_models_and_metrics()

model = all_artifacts['model_2']
scaler = all_artifacts['scaler']
feature_cols = all_artifacts['feature_cols']
prediction_accuracy = all_artifacts['accuracy_2']
prediction_auc = all_artifacts['auc_2']
threshold = 0.35

model_names = [
    'Random Forest (100)',
    'Random Forest (300) - Optimized',
    'Gradient Boosting',
    'Logistic Regression',
    'XGBoost/Fallback RF'
]
metrics_data = {
    'Model': model_names,
    'Accuracy': [all_artifacts[f'accuracy_{i+1}'] for i in range(5)],
    'AUC': [all_artifacts[f'auc_{i+1}'] for i in range(5)]
}
model_performance_df = pd.DataFrame(metrics_data)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 About This Tool")
    st.markdown(f"""
    <div class="sidebar-info">
    This dashboard predicts <b>30-day hospital readmission risk</b> for maternity patients using an
    <b>Optimized Random Forest</b> model.<br><br>
    <b>Features used:</b><br>
    • Age & Labor Duration<br>
    • Length of Stay<br>
    • Location (Urban/Rural)<br>
    • Complications (and engineered features)<br><br>
    <b>Model metrics (Optimized Random Forest):</b><br>
    • Accuracy: {prediction_accuracy:.1%}<br>
    • AUC-ROC: {prediction_auc:.3f}<br>
    • Risk threshold: {threshold:.0%}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚖️ Ethics Note")
    st.info("This tool is designed to **assist** clinical decision-making, not replace it. "
            "Always combine model output with clinical judgment.")

    st.markdown("---")
    st.markdown("### 📊 Fairness Audit")
    st.markdown("""
    **Delivery type gap:** ≤ 9.9% ✅ (Explicitly excluded to prevent discrimination)
    **Location gap:** Needs monitoring for potential bias ⚠
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
    st.markdown("**Medical Details**")
    complications = st.selectbox("Maternal Complications", ["No", "Yes"])
    labor_duration = st.slider("Labor Duration (hours)", min_value=1.0, max_value=24.0, value=10.0, step=0.5)

with col3:
    st.markdown("**Hospital Stay**")
    los = st.slider("Length of Stay (days)", min_value=2, max_value=14, value=4, step=1)
    st.markdown("<p style='font-size:0.8em; color:gray;'><i>*Delivery Type is handled internally for fairness.</i></p>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
if st.button("🔍 Calculate Readmission Risk"):

    location_encoded = 1 if location == 'Rural' else 0
    complications_encoded = 1 if complications == 'Yes' else 0

    complication_risk = complications_encoded * (los / 5)
    los_severity = los ** 1.5
    age_los_interaction = (age / 30) * los

    input_data_raw = pd.DataFrame([[
        age, labor_duration, los,
        location_encoded, complications_encoded,
        complication_risk, los_severity, age_los_interaction
    ]], columns=feature_cols)

    input_data_scaled = scaler.transform(input_data_raw)

    prob     = float(model.predict_proba(input_data_scaled)[0, 1])
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
        fig, ax = plt.subplots(figsize=(4.5, 3), subplot_kw={'aspect': 'equal'})
        theta_start, theta_end = np.pi, 0

        theta = np.linspace(theta_start, theta_end, 200)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=18, color='#E5E7EB', solid_capstyle='round')

        for zone_start, zone_end, color in [
            (np.pi, np.pi*2/3, '#10B981'),
            (np.pi*2/3, np.pi/3, '#FBBF24'),
            (np.pi/3, 0, '#EF4444'),
        ]:
            ztheta = np.linspace(zone_start, zone_end, 100)
            ax.plot(np.cos(ztheta), np.sin(ztheta), linewidth=18, color=color, alpha=0.3)

        filled_end = np.pi - prob * np.pi
        vtheta = np.linspace(theta_start, filled_end, 200)
        arc_color = '#EF4444' if is_high else '#10B981'
        ax.plot(np.cos(vtheta), np.sin(vtheta), linewidth=18, color=arc_color, solid_capstyle='round')

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
        st.pyplot(fig)
        plt.close()

    with factors_col:
        st.markdown("**Risk Factor Summary**")

        factor_items = [
            ("Age", age, 30, "younger", "older"),
            ("LOS", los, 5, "shorter", "longer"),
            ("Labor Dur.", labor_duration, 12, "shorter", "longer"),
        ]

        for name, val, ref, low_label, high_label in factor_items:
            direction = "↑ Higher risk" if val > ref else "↓ Lower risk"
            color = "#EF4444" if val > ref else "#10B981"
            st.markdown(f"**{name}:** {val} → <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)

        st.markdown("---")
        flag_style = lambda cond: "🔴" if cond else "🟢"
        st.markdown(f"{flag_style(location=='Rural')} **{location}** location")
        st.markdown(f"{flag_style(complications=='Yes')} Complications: **{complications}**")

    st.markdown("---")
    st.markdown("### 💊 Clinical Recommendations")
    if is_high:
        rcols = st.columns(3)
        with rcols[0]:
            st.error("**☎️ Schedule Follow-up**\nBook appointment within 7 days of discharge")
        with rcols[1]:
            st.warning("**📋 Enhanced Discharge Plan**\nProvide written instructions for warning signs")
        with rcols[2]:
            st.info("**🩺 Home Nursing**\nConsider home nursing visit within 48–72 hours")
    else:
        rcols = st.columns(3)
        with rcols[0]:
            st.success("**📅 Standard Follow-up**\nRoutine 6-week postnatal appointment")
        with rcols[1]:
            st.success("**☎️ Helpline**\nProvide helpline number for questions")
        with rcols[2]:
            st.success("**📱 Patient App**\nEnroll in standard maternity tracking app")

    st.caption(f"⚠️ This tool is for clinical decision support only. Risk threshold: {threshold:.0%}. "
               f"All decisions should be made by qualified medical professionals.")


# ── Batch / Population Analysis ───────────────────────────────
st.markdown("---")
st.markdown("### 📊 Population Analysis (Demo)")

demo_df = pd.DataFrame({
    'Age':           [22, 35, 40, 28, 33, 41, 19, 38, 45, 25],
    'LaborDuration': [8,  20,  5, 15, 12,  3, 18, 22,  6, 10],
    # FIX 1: renamed LOS → LengthofStaydays to match feature_cols
    'LengthofStaydays': [3, 7, 4, 5, 6, 8, 3, 9, 5, 4],
    'DeliveryType':  ['Vaginal','Cesarean','Vaginal','Cesarean','Vaginal',
                      'Cesarean','Vaginal','Cesarean','Vaginal','Vaginal'],
    # FIX 2: corrected typo 'Cesarean' → 'Urban' in Location column
    'Location':      ['Urban','Rural','Urban','Rural','Urban','Urban',
                      'Urban','Urban','Rural','Urban'],
    'Complications': ['No','Yes','No','Yes','No','Yes','No','Yes','No','No'],
})

demo_df['Location_Encoded']    = demo_df['Location'].apply(lambda x: 1 if x == 'Rural' else 0)
demo_df['Complications_Encoded'] = demo_df['Complications'].apply(lambda x: 1 if x == 'Yes' else 0)
demo_df['Complication_Risk']   = demo_df['Complications_Encoded'] * (demo_df['LengthofStaydays'] / 5)
demo_df['LOS_Severity']        = demo_df['LengthofStaydays'] ** 1.5
demo_df['Age_LOS_Interaction'] = (demo_df['Age'] / 30) * demo_df['LengthofStaydays']

X_demo        = demo_df[feature_cols]          # now works: all cols present
X_demo_scaled = scaler.transform(X_demo)
demo_df['Risk_Prob']  = model.predict_proba(X_demo_scaled)[:, 1]
demo_df['Risk_Level'] = demo_df['Risk_Prob'].apply(lambda p: 'HIGH' if p >= threshold else 'LOW')
demo_df['Risk_Prob']  = (demo_df['Risk_Prob'] * 100).round(1).astype(str) + '%'

def color_risk(val):
    if val == 'HIGH':
        return 'background-color: #FEE2E2; color: #DC2626; font-weight: bold'
    return 'background-color: #D1FAE5; color: #059669; font-weight: bold'

# FIX 3: display LengthofStaydays column (renamed from LOS) and use use_container_width instead of width='stretch'
styled = demo_df[['Age', 'LaborDuration', 'LengthofStaydays', 'Location',
                   'Complications', 'Risk_Prob', 'Risk_Level']].style.map(color_risk, subset=['Risk_Level'])
st.dataframe(styled, use_container_width=True)


# ── Educational Tabs ──────────────────────────────────────────
# FIX 4: corrected broken emoji in tab1 label
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Feature Impact",
    "📊 Model Performance",
    "⚖️ Ethics & Fairness",
    "❓ FAQ"
])

with tab1:
    st.subheader("Feature Importance in Risk Prediction")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_cols))

    readable_feature_names = {
        'Age': 'Age', 'LaborDuration': 'Labor Duration', 'LengthofStaydays': 'Length of Stay',
        'Location_Encoded': 'Location (Rural)', 'Complications_Encoded': 'Complications (Yes)',
        'Complication_Risk': 'Complication Risk', 'LOS_Severity': 'LOS Severity',
        'Age_LOS_Interaction': 'Age-LOS Interaction'
    }
    feature_names_for_plot = [readable_feature_names.get(col, col.replace('_', ' ').title()) for col in feature_cols]

    importance_df_plot = pd.DataFrame({
        'Feature': feature_names_for_plot,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df_plot['Feature'], importance_df_plot['Importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Feature Importance in Readmission Prediction ({type(model).__name__})')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Key Insights from the Primary Model:**
    - **Age-LOS Interaction**: Combines patient age and length of stay — strong influence on readmission risk.
    - **Length of Stay & LOS Severity**: Longer stays and their non-linear severity are crucial indicators.
    - **Labor Duration**: Reflects the intensity and complexity of the delivery process.
    - **Complication Risk**: Combines complications with length of stay — a direct clinical risk factor.
    - **Age**: A fundamental demographic risk factor.
    - **Location & Complications**: Direct indicators contributing to risk assessment.
    """)

with tab2:
    st.subheader("Model Performance Metrics Overview")
    st.markdown("Comparative accuracy and AUC scores for all trained models.")
    st.dataframe(model_performance_df.round(3), use_container_width=True)

    st.markdown(f"""
    **Training Dataset Details:**
    - ~463 patients (after quality checks)
    - ~25.3% readmission rate
    - Train-test split: 80-20
    - **Primary Model**: Optimized Random Forest (300 trees) selected for its balanced performance.
    """)

with tab3:
    st.subheader("⚖️ Ethical Fairness Considerations")

    st.markdown("""
    ### Fairness Principle: Individual Fairness
    **Definition:** Similar patients (by clinical measures) receive similar risk assessments,
    regardless of demographic characteristics.

    ### Design Choices

    ✅ **Included Features:**
    - Age, Labor Duration, Length of Stay, Complications, Location, and engineered combinations.
    - All chosen for their **clinical relevance and causal relationship** with readmission risk.

    ❌ **Excluded Features:**
    - **Delivery Type (Vaginal vs. Cesarean)**: Could perpetuate historical biases; underlying clinical
      reasons (e.g., complications) are already captured by other features.

    ### Bias Monitoring
    - Performance audited across subgroups (location, age groups) quarterly.
    - `class_weight='balanced'` mitigates majority-class bias in the primary model.
    - Results available in the ethics audit report.

    ### Limitations
    - Model trained on hospital data; may not generalise to all settings.
    - Clinical validation required before deployment.
    - Always requires human oversight and clinical judgment.
    - Patients have the right to know their risk prediction and its rationale.

    ### ICMR Compliance (Illustrative)
    - ✓ Data privacy (de-identified training data)
    - ✓ Informed consent (assumed within healthcare system)
    - ✓ Fairness audits (as described above)
    - ⚠️ IRB approval and institutional validation crucial before clinical deployment.
    """)

with tab4:
    st.subheader("Frequently Asked Questions")

    with st.expander("1. What is the model predicting?"):
        st.write("""
        The model predicts the probability that a maternity patient will be readmitted
        within 30 days of discharge, helping identify patients who need more intensive follow-up.
        """)

    with st.expander("2. Why doesn't the model include delivery type?"):
        st.write("""
        Delivery type is statistically associated with readmission but may introduce fairness biases.
        The model instead focuses on the underlying clinical factors (Complications, Length of Stay)
        which are the more direct drivers of readmission, ensuring a more equitable assessment.
        """)

    with st.expander("3. How accurate is the primary prediction model?"):
        st.write(f"""
        The primary model (Optimized Random Forest) achieves an accuracy of **{prediction_accuracy:.1%}**
        and an AUC score of **{prediction_auc:.3f}** on test data. Always validate within your
        specific hospital setting before clinical deployment.
        """)

    with st.expander("4. Can I override the model's prediction?"):
        st.write("""
        Absolutely. This is a **decision support tool only**. Clinical judgment always takes
        precedence. Final follow-up decisions must be made by qualified medical professionals.
        """)

    with st.expander("5. What should I do with a high-risk prediction?"):
        st.write("""
        High-risk predictions suggest enhanced care such as:
        - Early follow-up within 24–48 hours post-discharge.
        - Detailed assessment of warning signs and patient needs.
        - Clear, actionable discharge instructions.
        - Access to on-call support or dedicated helplines.
        - Coordinated care planning with home health or specialist services.
        """)

    with st.expander("6. How often is the model updated?"):
        st.write("""
        Best practice: retrain and validate annually with fresh patient data to remain accurate
        as populations and clinical practices evolve. Regular fairness audits should be included.
        """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🏥 Maternity Readmission Risk Prediction System</p>
    <p>Developed with focus on fairness, transparency, and clinical validity</p>
    <p style='margin-top: 10px;'><small>Disclaimer: This tool is for educational and research purposes.
    For clinical use, obtain IRB approval and institutional validation. Always consult clinical judgment.</small></p>
</div>
""", unsafe_allow_html=True)
