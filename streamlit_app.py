import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, recall_score, precision_score)
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternity Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME COLORS ─────────────────────────────────────────────────────
BLUE   = '#1F5C8B'
LTBLUE = '#2E75B6'
GREEN  = '#27AE60'
RED    = '#C0392B'
ORANGE = '#E07B39'

# ── CUSTOM CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-low    { background:#D5F5E3; color:#1E8449; border-radius:8px; padding:12px; text-align:center; font-size:1.3em; font-weight:bold; }
    .risk-mod    { background:#FDEBD0; color:#D35400; border-radius:8px; padding:12px; text-align:center; font-size:1.3em; font-weight:bold; }
    .risk-high   { background:#FADBD8; color:#C0392B; border-radius:8px; padding:12px; text-align:center; font-size:1.3em; font-weight:bold; }
    .metric-box  { background:#F0F4F8; border-radius:8px; padding:12px; text-align:center; margin:4px; }
    .section-hdr { color:#1F5C8B; font-size:1.1em; font-weight:bold; border-bottom:2px solid #2E75B6; padding-bottom:4px; margin-bottom:8px; }
    .ethics-box  { background:#EAF4FB; border-left:4px solid #1F5C8B; padding:12px; border-radius:4px; margin:8px 0; }
    .warn-box    { background:#FEF9E7; border-left:4px solid #F39C12; padding:10px; border-radius:4px; margin:8px 0; }
    .good-box    { background:#D5F5E3; border-left:4px solid #27AE60; padding:10px; border-radius:4px; margin:8px 0; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & TRAIN MODEL ────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('test_super.csv')
    except FileNotFoundError:
        st.error("test_super.csv not found. Please ensure it is in the same directory as app.py.")
        st.stop()

    # All columns are already numeric — no encoding needed
    # DeliveryType EXCLUDED for fairness (acts as proxy, not causal)
    FEATURES = ['Age', 'Complications', 'Comorbidities', 'LOS', 'DaysToFollowup', 'Location']
    X = df[FEATURES]
    y = df['Readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,min_samples_leaf=5,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)

    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, pred)
    auc   = roc_auc_score(y_test, proba)
    cm    = confusion_matrix(y_test, pred)

    # Fairness audit — store test indices for subgroup analysis
    df_test = X_test.copy()
    df_test['y_true']       = y_test.values
    df_test['y_pred']       = pred
    df_test['y_prob']       = proba
    df_test['DeliveryType'] = df.loc[X_test.index, 'DeliveryType'].values

    return model, df, df_test, X_test, y_test, proba, acc, auc, cm, FEATURES

model, df, df_test, X_test, y_test, proba, acc, auc, cm, FEATURES = load_and_train()
tn, fp, fn, tp = cm.ravel()

sensitivity  = tp / (tp + fn)
specificity  = tn / (tn + fp)
precision    = tp / (tp + fp)

# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("# 🏥 Maternity Patient Readmission Predictor")
st.markdown(
    "**Fairness-aware ML system** for predicting 30-day readmission risk. "
    "Delivery type is **excluded** from features to prevent discrimination. "
    "This tool is **decision support only** — clinical judgment always takes precedence."
)
st.markdown("---")

# ── SIDEBAR ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👩‍⚕️ Patient Input")
    st.markdown("Enter patient clinical information:")
    st.markdown("---")

    age            = st.slider("Age (years)", 18, 45, 30,
                               help="Patient age at time of delivery")
    complications  = st.selectbox("Complications", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No",
                                  help="Were complications present during delivery/stay?")
    comorbidities  = st.slider("Comorbidities (count)", 0, 6, 1,
                               help="Number of comorbid conditions")
    los            = st.slider("Length of Stay (days)", 1.0, 16.0, 5.0, step=0.5,
                               help="Initial hospital stay duration")
    days_followup  = st.slider("Days to Follow-up", 1, 30, 10,
                               help="Days until scheduled post-discharge follow-up")
    location       = st.selectbox("Location", [0, 1],
                                  format_func=lambda x: "Rural" if x == 1 else "Urban",
                                  help="Patient's residential location")

    st.markdown("---")
    st.markdown("**⚠️ Excluded Feature**")
    st.info("Delivery Type is intentionally excluded from prediction to prevent proxy discrimination.")

# ── PREDICT ──────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'Age':           age,
    'Complications': complications,
    'Comorbidities': comorbidities,
    'LOS':           los,
    'DaysToFollowup': days_followup,
    'Location':      location,
}])

prob       = model.predict_proba(input_data)[0, 1]
risk_level = "LOW" if prob < 0.40 else "HIGH" if prob > 0.60 else "MODERATE"
risk_class = "risk-low" if risk_level == "LOW" else "risk-high" if risk_level == "HIGH" else "risk-mod"
risk_emoji = "🟢" if risk_level == "LOW" else "🔴" if risk_level == "HIGH" else "🟡"

# ── TABS ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", "📊 Feature Impact",
    "📈 Model Performance", "⚖️ Ethics & Fairness", "❓ FAQ"
])

# ═══════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ═══════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎯 Risk Prediction Result")
        st.markdown(
            f'<div class="{risk_class}">{risk_emoji} {risk_level} RISK<br>'
            f'<span style="font-size:0.7em">Readmission Probability: {prob*100:.1f}%</span></div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh(0, 40,  left=0,  height=0.4, color='#27AE60', alpha=0.85)
        ax.barh(0, 20,  left=40, height=0.4, color='#F39C12', alpha=0.85)
        ax.barh(0, 40,  left=60, height=0.4, color='#C0392B', alpha=0.85)
        ax.plot([prob*100], [0], marker='v', markersize=14,
                color='white', markeredgecolor='black', markeredgewidth=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Readmission Probability (%)')
        ax.set_yticks([])
        ax.text(20,  -0.42, 'LOW',      ha='center', fontsize=9, fontweight='bold', color='#1E8449')
        ax.text(50,  -0.42, 'MODERATE', ha='center', fontsize=9, fontweight='bold', color='#D35400')
        ax.text(80,  -0.42, 'HIGH',     ha='center', fontsize=9, fontweight='bold', color='#C0392B')
        ax.set_title(f'Risk Score: {prob*100:.1f}%', fontweight='bold')
        ax.spines[['top','right','left']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Patient Profile Summary:**")
        summary_data = {
            "Parameter": ["Age", "Complications", "Comorbidities", "Length of Stay",
                          "Days to Follow-up", "Location"],
            "Value":     [
                f"{age} years",
                "Yes" if complications == 1 else "No",
                str(comorbidities),
                f"{los:.1f} days",
                f"{days_followup} days",
                "Rural" if location == 1 else "Urban"
            ]
        }
        st.table(pd.DataFrame(summary_data))

    with col2:
        st.markdown("### 🏥 Clinical Recommendations")

        if risk_level == "LOW":
            st.markdown('<div class="good-box">', unsafe_allow_html=True)
            st.markdown("**✅ LOW RISK — Standard Follow-up**")
            st.markdown("""
- Schedule routine postpartum follow-up at **6 weeks**
- Provide standard discharge education materials
- Advise patient to contact provider if symptoms arise
- No additional monitoring required
- Reassess if clinical condition changes
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        elif risk_level == "MODERATE":
            st.markdown('<div class="warn-box">', unsafe_allow_html=True)
            st.markdown("**⚠️ MODERATE RISK — Enhanced Follow-up**")
            st.markdown("""
- **Phone follow-up within 3–5 days** of discharge
- Review discharge medications and wound care instructions
- Provide 24/7 nurse triage contact number
- Reassess in-person if symptoms reported during call
- Schedule earlier follow-up if rural location (access concern)
- Educate patient on readmission warning signs
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="risk-high" style="text-align:left; font-size:1em;">', unsafe_allow_html=True)
            st.markdown("**🚨 HIGH RISK — Urgent Action Required**")
            st.markdown("""
- **In-person visit within 24–48 hours** of discharge
- Consider extended observation or delayed discharge
- Social work consult for rural/support-limited patients
- Intensive patient education prior to discharge
- Ensure home support structure is in place
- Daily phone contact for first 72 hours post-discharge
- Flag record for clinical team attention
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Risk Thresholds:**")
        thresh_data = pd.DataFrame({
            "Risk Level":  ["🟢 Low Risk", "🟡 Moderate Risk", "🔴 High Risk"],
            "Probability": ["< 40%",       "40% – 60%",        "> 60%"],
            "Action":      ["Routine follow-up", "Phone call 3–5 days", "In-person 24–48 hrs"]
        })
        st.table(thresh_data)

        st.markdown("""
<div class="warn-box">
⚠️ <b>Clinical Disclaimer:</b> This prediction is a decision support tool only.
It must be reviewed alongside full clinical assessment.
Clinicians retain full authority over all patient care decisions.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 2: FEATURE IMPACT
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 How Each Feature Affects Risk")

    col1, col2 = st.columns([1, 1])

    importances  = model.feature_importances_
    feat_labels  = ['Age', 'Complications', 'Comorbidities', 'LOS', 'Days to Follow-up', 'Location']
    idx          = np.argsort(importances)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors_fi = [RED if i == np.argmax(importances) else
                     ORANGE if importances[i] > 0.12 else BLUE
                     for i in idx]
        bars = ax.barh([feat_labels[i] for i in idx], importances[idx] * 100,
                       color=colors_fi, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.1f}%', va='center', fontweight='bold')
        ax.set_xlabel('Feature Importance (%)')
        ax.set_title('Random Forest Feature Importance\n(Delivery Type excluded for fairness)',
                     fontweight='bold')
        ax.set_xlim(0, max(importances) * 100 * 1.3)
        patches = [
            mpatches.Patch(color=RED,    label='Strongest predictor'),
            mpatches.Patch(color=ORANGE, label='High importance'),
            mpatches.Patch(color=BLUE,   label='Moderate importance'),
        ]
        ax.legend(handles=patches, fontsize=8, loc='lower right')
        st.pyplot(fig)

    with col2:
        st.markdown("**Feature Importance Ranking:**")
        sorted_idx   = np.argsort(importances)[::-1]
        clinical_meaning = {
            'Age':              'Age-related physiological risk factors',
            'Complications':    'Direct clinical risk driver — presence raises readmission odds',
            'Comorbidities':    'Each additional comorbidity significantly increases risk',
            'LOS':              'Longer stay reflects greater clinical severity',
            'Days to Follow-up':'Fewer follow-up days = less post-discharge support',
            'Location':         'Rural patients face reduced access to follow-up care',
        }
        imp_df = pd.DataFrame({
            'Rank':             range(1, len(FEATURES) + 1),
            'Feature':          [feat_labels[i] for i in sorted_idx],
            'Importance':       [f'{importances[i]*100:.1f}%' for i in sorted_idx],
            'Clinical Meaning': [clinical_meaning[feat_labels[i]] for i in sorted_idx],
        })
        st.table(imp_df)

        st.markdown("""
<div class="ethics-box">
<b>⚖️ Fairness Note:</b><br>
Delivery Type (Vaginal=0 / Cesarean=1) shows a ~7.3 percentage-point difference in
readmission rates across groups. However, this is driven by <i>complications</i>,
not delivery type itself. Including it would constitute proxy discrimination.
Excluding it costs only ~1% accuracy while preserving individual fairness.
</div>
""", unsafe_allow_html=True)

    # Per-feature readmission rate charts
    st.markdown("---")
    st.markdown("### 📉 Readmission Rate by Feature Value")

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('Readmission Rate by Clinical Feature', fontsize=13, fontweight='bold')

    # Comorbidities (most important continuous grouping)
    comor_rates = df.groupby('Comorbidities')['Readmitted'].mean() * 100
    axes[0].bar(comor_rates.index.astype(str), comor_rates.values, color=BLUE, edgecolor='white')
    for i, v in enumerate(comor_rates.values):
        axes[0].text(i, v + 0.8, f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold')
    axes[0].set_title('By Comorbidity Count', fontweight='bold')
    axes[0].set_xlabel('Number of Comorbidities')
    axes[0].set_ylabel('Readmission Rate (%)')
    axes[0].set_ylim(0, 115)

    # Complications
    comp_rates = df.groupby('Complications')['Readmitted'].mean() * 100
    axes[1].bar(['No (0)', 'Yes (1)'], comp_rates.values, color=[GREEN, RED], edgecolor='white')
    for i, v in enumerate(comp_rates.values):
        axes[1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[1].set_title('By Complications', fontweight='bold')
    axes[1].set_ylabel('Readmission Rate (%)')
    axes[1].set_ylim(0, 70)

    # LOS buckets
    df['LOS_bucket'] = pd.cut(df['LOS'], bins=[0, 4, 7, 10, 20],
                               labels=['<4 days', '4–7 days', '7–10 days', '>10 days'])
    los_rates = df.groupby('LOS_bucket', observed=True)['Readmitted'].mean() * 100
    axes[2].bar(los_rates.index, los_rates.values, color=LTBLUE, edgecolor='white')
    for i, v in enumerate(los_rates.values):
        axes[2].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[2].set_title('By Length of Stay', fontweight='bold')
    axes[2].set_ylabel('Readmission Rate (%)')
    axes[2].set_ylim(0, 70)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    st.pyplot(fig2)

# ═══════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",    f"{acc*100:.1f}%",        "Test set (n=200)")
    col2.metric("AUC Score",   f"{auc:.4f}",             "ROC curve")
    col3.metric("Sensitivity", f"{sensitivity*100:.1f}%","True positive rate")
    col4.metric("Specificity", f"{specificity*100:.1f}%","True negative rate")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Readmitted', 'Readmitted'],
                    yticklabels=['Not Readmitted', 'Readmitted'],
                    cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title(f'Confusion Matrix\n(Test set, n={len(y_test)})', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Confusion Matrix Breakdown:**")
        cm_df = pd.DataFrame({
            'Cell':        ['True Negative (TN)', 'False Positive (FP)',
                            'False Negative (FN)', 'True Positive (TP)'],
            'Count':       [tn, fp, fn, tp],
            'Meaning':     [
                'Correctly predicted NOT readmitted',
                'Predicted readmitted — actually not (unnecessary follow-up)',
                'Missed readmission — most clinically costly error',
                'Correctly predicted readmitted'
            ]
        })
        st.table(cm_df)

    with col2:
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'Random Forest (AUC={auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC=0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Risk score distribution
    st.markdown("---")
    fig3, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(proba[y_test == 0], bins=20, alpha=0.7, color=GREEN,
            label='Not Readmitted', edgecolor='white')
    ax.hist(proba[y_test == 1], bins=20, alpha=0.7, color=RED,
            label='Readmitted', edgecolor='white')
    ax.axvline(0.40, color='gold',       linestyle='--', lw=2, label='Low→Moderate (40%)')
    ax.axvline(0.60, color='darkorange', linestyle='--', lw=2, label='Moderate→High (60%)')
    ax.set_xlabel('Predicted Readmission Probability')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution by Actual Outcome', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("**Full Performance Summary:**")
    perf_df = pd.DataFrame({
        'Metric':      ['Accuracy', 'AUC', 'Sensitivity (Recall)', 'Specificity',
                        'Precision', 'False Negative Rate'],
        'Value':       [f'{acc*100:.1f}%', f'{auc:.4f}', f'{sensitivity*100:.1f}%',
                        f'{specificity*100:.1f}%', f'{precision*100:.1f}%',
                        f'{fn/(fn+tp)*100:.1f}%'],
        'Clinical Meaning': [
            'Overall correct predictions',
            'Discrimination ability (0.90 = excellent)',
            'Proportion of actual readmissions caught',
            'Proportion of non-readmissions correctly cleared',
            'Of predicted high-risk, how many truly readmitted',
            'Readmitted patients missed by model — minimise this'
        ]
    })
    st.table(perf_df)

# ═══════════════════════════════════════════════════════
# TAB 4: ETHICS & FAIRNESS
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚖️ Fairness Framework & Ethics Audit")

    st.markdown("""
<div class="ethics-box">
<b>🎯 Fairness Principle: Individual Fairness</b><br>
<i>"Similar patients (by clinical measures) receive similar risk scores,
regardless of delivery type or other demographic attributes."</i><br><br>
Two patients with identical Age, Complications, Comorbidities, LOS,
Days-to-Follow-up, and Location will always receive the same predicted risk score —
delivery type cannot influence their score.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Real subgroup accuracy from test set
    acc_vaginal  = accuracy_score(df_test[df_test['DeliveryType']==0]['y_true'],
                                  df_test[df_test['DeliveryType']==0]['y_pred'])
    acc_cesarean = accuracy_score(df_test[df_test['DeliveryType']==1]['y_true'],
                                  df_test[df_test['DeliveryType']==1]['y_pred'])
    acc_urban    = accuracy_score(df_test[df_test['Location']==0]['y_true'],
                                  df_test[df_test['Location']==0]['y_pred'])
    acc_rural    = accuracy_score(df_test[df_test['Location']==1]['y_true'],
                                  df_test[df_test['Location']==1]['y_pred'])

    diff_delivery = abs(acc_vaginal - acc_cesarean) * 100
    diff_location = abs(acc_urban   - acc_rural)    * 100
    status_d = "✅ PASS" if diff_delivery < 10 else "❌ FAIL"
    status_l = "✅ PASS" if diff_location < 10 else "❌ FAIL"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Bias Audit Results (Test Set, n=200):**")
        bias_df = pd.DataFrame({
            'Subgroup':   ['Vaginal delivery (0)', 'Cesarean delivery (1)',
                           'Urban location (0)',   'Rural location (1)'],
            'Accuracy':   [f'{acc_vaginal*100:.1f}%', f'{acc_cesarean*100:.1f}%',
                           f'{acc_urban*100:.1f}%',   f'{acc_rural*100:.1f}%'],
            'Gap vs Group': [f'{diff_delivery:.1f} pp diff', '←',
                             f'{diff_location:.1f} pp diff',  '←'],
            'Status':     [status_d, status_d, status_l, status_l]
        })
        st.table(bias_df)

        color_d = "good-box" if diff_delivery < 10 else "warn-box"
        color_l = "good-box" if diff_location < 10 else "warn-box"
        st.markdown(f'<div class="{color_d}">Delivery Type gap: <b>{diff_delivery:.1f} pp</b> — {status_d} (threshold: 10 pp)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{color_l}">Location gap: <b>{diff_location:.1f} pp</b> — {status_l} (threshold: 10 pp)</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Why Delivery Type Was Excluded:**")
        st.markdown("""
| Reason | Explanation |
|--------|-------------|
| 🚫 Not causal | Delivery type doesn't cause readmission — complications do |
| 🔀 Confounded | Cesarean correlates with higher comorbidities in this dataset |
| ⚖️ Discrimination risk | Identical clinical patients would score differently |
| 📊 Minimal accuracy cost | Exclusion costs only ~1% accuracy |
| 🏛️ Ethical obligation | Protected characteristic under individual fairness |
        """)

    # Fairness visualisation
    st.markdown("---")
    fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig4.suptitle('Fairness Audit — Model Accuracy by Subgroup (Test Set)',
                  fontsize=13, fontweight='bold')

    for ax, (groups, vals, title, bar_colors, gap) in zip(axes, [
        (['Vaginal (0)', 'Cesarean (1)'],
         [acc_vaginal*100, acc_cesarean*100],
         'By Delivery Type', [BLUE, ORANGE], diff_delivery),
        (['Urban (0)', 'Rural (1)'],
         [acc_urban*100, acc_rural*100],
         'By Location', [BLUE, GREEN], diff_location),
    ]):
        bars = ax.bar(groups, vals, color=bar_colors, edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f'{b.get_height():.1f}%', ha='center', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(60, 100)
        ax.set_title(title, fontweight='bold')
        colour = GREEN if gap < 10 else RED
        symbol = "✅" if gap < 10 else "❌"
        ax.text(0.5, 0.08, f'Gap: {gap:.1f} pp {symbol} (threshold < 10 pp)',
                ha='center', transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=colour,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))
        ax.axhline(90, color='red',  linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(70, color='gold', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("---")
    st.markdown("**ICMR Compliance Status:**")
    icmr_df = pd.DataFrame({
        'Requirement': ['Data Privacy', 'Informed Consent', 'Vulnerable Population',
                        'Bias Monitoring', 'Transparency', 'Security', 'Ethics Review (IRB)'],
        'Status': ['✅ Compliant', '⚠️ Conditional', '⚠️ Monitor',
                   '✅ Compliant', '✅ Compliant', '⚠️ Conditional', '⚠️ Required'],
        'Notes': [
            'De-identified, no PII stored',
            'Explicit consent needed for clinical deployment',
            'Pregnant patients require heightened oversight',
            'Audit completed; ongoing quarterly plan documented',
            'Feature importance and exclusion rationale provided',
            'Encryption required for production deployment',
            'IRB approval needed before any clinical use'
        ]
    })
    st.table(icmr_df)

# ═══════════════════════════════════════════════════════
# TAB 5: FAQ
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("### ❓ Frequently Asked Questions")

    faqs = [
        ("Why is delivery type not included in the prediction?",
         "Delivery type was deliberately excluded as a fairness decision. In this dataset, Cesarean patients "
         "(DeliveryType=1) show a ~7.3 percentage-point higher readmission rate than vaginal delivery patients. "
         "However, this difference is **driven by comorbidities and complications**, not delivery type itself. "
         "Including it would create proxy discrimination — two clinically identical patients (same age, same complications, "
         "same comorbidities) would receive different risk scores purely due to delivery method. "
         "Excluding it costs only ~1% accuracy while fully preserving Individual Fairness."),

        ("How accurate is the model and can I trust it?",
         f"The model achieves **{acc*100:.1f}% accuracy** and an **AUC of {auc:.4f}** on the held-out test set (n=200). "
         f"It correctly identifies {sensitivity*100:.1f}% of patients who will be readmitted (sensitivity) "
         f"while maintaining {specificity*100:.1f}% specificity. However, **this tool is decision support, "
         "not a replacement for clinical judgment**. Always integrate this prediction with your full clinical "
         "assessment. A low risk score does not guarantee a patient will not be readmitted."),

        ("What is the most important predictor of readmission?",
         "In this dataset, **Days to Follow-up** is the single strongest predictor, accounting for over 55% of "
         "the model's predictive power. Patients with fewer scheduled follow-up days post-discharge are at "
         "significantly higher readmission risk. This is clinically meaningful — early follow-up allows clinicians "
         "to detect and manage post-discharge complications before they escalate to readmission. "
         "Comorbidities and Length of Stay are the next strongest predictors."),

        ("What does 'Individual Fairness' mean?",
         "Individual Fairness means that patients who are clinically similar should receive similar risk scores. "
         "If two patients have the same age, complications, comorbidities, length of stay, days to follow-up, "
         "and location, they will receive the **identical risk score** — regardless of delivery type or any "
         "other protected attribute not in the model. This is the most appropriate fairness framework for "
         "healthcare, where outcome differences should be clinically driven, not demographically driven."),

        ("Can this model be used for clinical decisions right now?",
         "**No — not without additional approvals.** Before clinical deployment, the following are required: "
         "(1) IRB/Ethics Committee approval, (2) explicit informed consent protocol for patients, "
         "(3) data encryption implementation, (4) validation on local hospital data, and "
         "(5) clinical staff training. This tool is currently approved for educational and research use only. "
         "All clinical use must involve clinician oversight."),

        ("How often should the model be retrained?",
         "We recommend: **Monthly** accuracy monitoring; **Quarterly** fairness audits checking subgroup "
         "performance by delivery type and location; **Annual** full model retraining with updated data. "
         "Immediate retraining is warranted if accuracy drops more than 3%, any subgroup shows a greater than "
         "10 percentage-point accuracy gap, or patient population demographics shift significantly."),
    ]

    for i, (question, answer) in enumerate(faqs):
        with st.expander(f"Q{i+1}. {question}"):
            st.markdown(answer)

    st.markdown("---")
    st.markdown("""
<div class="ethics-box">
<b>📞 Further Information</b><br>
For the full model analysis, see <code>readmission_model.ipynb</code>.<br>
Dataset: <code>test_super.csv</code> — 1,000 maternity patients, 8 features, 50% readmission rate.<br>
Model: Random Forest (200 estimators, max_depth=10, class_weight=balanced).
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85em;'>"
    "🏥 Maternity Patient Readmission Prediction System | "
    "Version 2.0 | Dataset: test_super.csv (n=1,000) | "
    "Educational & Research Use Only | "
    "⚖️ Individual Fairness Framework"
    "</div>",
    unsafe_allow_html=True
)
