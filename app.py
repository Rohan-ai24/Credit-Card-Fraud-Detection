import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             f1_score, ConfusionMatrixDisplay)
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Bebas+Neue&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }

.stApp { background-color: #080b10; color: #c8cdd6; }

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1a2030;
}

[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1a2030;
    border-radius: 8px;
    padding: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 8px;
    gap: 2px;
    border: 1px solid #1a2030;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a6475;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    background: #1a2030 !important;
    color: #ff3c3c !important;
}

.stButton > button {
    background: #ff3c3c;
    color: #080b10;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 24px;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; color: #080b10; }

label { color: #5a6475 !important; font-size: 12px !important; font-family: 'IBM Plex Mono', monospace !important; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.08em; color: #e8eaf0 !important; }
h1 { font-size: 2.4rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.3rem !important; }

.fraud-box {
    border-radius: 8px;
    padding: 22px 26px;
    margin-top: 16px;
    font-family: 'IBM Plex Mono', monospace;
}
.fraud-alert  { background: #1a0a0a; border: 1.5px solid #ff3c3c; color: #ff8080; }
.fraud-safe   { background: #0a1a0f; border: 1.5px solid #22c55e; color: #6ee7a0; }

.alert-badge {
    display: inline-block;
    background: #ff3c3c;
    color: #080b10;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

hr { border-color: #1a2030; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#080b10",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#1a2030",
    "axes.labelcolor":   "#5a6475",
    "xtick.color":       "#5a6475",
    "ytick.color":       "#5a6475",
    "text.color":        "#c8cdd6",
    "grid.color":        "#1a2030",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         10,
})
RED     = "#ff3c3c"
GREEN   = "#22c55e"
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
MUTED   = "#1e2535"

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    df_clean = df.copy()
    scaler = StandardScaler()
    df_clean['Amount'] = scaler.fit_transform(df_clean[['Amount']])
    df_clean['Time']   = scaler.fit_transform(df_clean[['Time']])
    X = df_clean.drop('Class', axis=1)
    y = df_clean['Class']
    return df, X, y, scaler

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # Logistic Regression
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, sampling_strategy=0.3)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_train, y_train)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_tr_sm, y_tr_sm)
    p = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {'model': lr, 'probs': p,
        'preds': lr.predict(X_test), 'auc': roc_auc_score(y_test, p),
        'ap': average_precision_score(y_test, p)}

    # Balanced Random Forest
    rf = BalancedRandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {'model': rf, 'probs': p,
        'preds': rf.predict(X_test), 'auc': roc_auc_score(y_test, p),
        'ap': average_precision_score(y_test, p)}

    # XGBoost
    fw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=fw, subsample=0.8, colsample_bytree=0.8,
        eval_metric='aucpr', random_state=42)
    xgb_m.fit(X_train, y_train)
    p = xgb_m.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {'model': xgb_m, 'probs': p,
        'preds': xgb_m.predict(X_test), 'auc': roc_auc_score(y_test, p),
        'ap': average_precision_score(y_test, p)}

    # Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.002,
                          random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_raw = iso.predict(X_test)
    iso_preds = (iso_raw == -1).astype(int)
    iso_scores = -iso.score_samples(X_test)
    results['Isolation Forest'] = {'model': iso, 'probs': iso_scores,
        'preds': iso_preds, 'auc': roc_auc_score(y_test, iso_scores),
        'ap': average_precision_score(y_test, iso_scores)}

    # Best threshold for XGBoost
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, (results['XGBoost']['probs'] >= t).astype(int))
           for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]

    feat_imp = pd.Series(xgb_m.feature_importances_,
                         index=X.columns).sort_values(ascending=False)

    return results, X_test, y_test, best_thresh, feat_imp, X_train, y_train

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔴 FRAUD DETECTION")
    st.markdown("---")
    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])
    st.markdown("---")
    st.caption("Models trained on upload:")
    st.caption("• Logistic Regression")
    st.caption("• Balanced Random Forest")
    st.caption("• XGBoost (scale_pos_weight)")
    st.caption("• Isolation Forest (anomaly)")
    st.markdown("---")
    st.caption("Dataset: Kaggle Credit Card Fraud\n284,807 transactions · 492 frauds")

if uploaded is None:
    st.markdown("# 🔴 FRAUD DETECTION SYSTEM")
    st.info("👈 Upload **creditcard.csv** in the sidebar to begin.\n\nDownload from: kaggle.com/datasets/mlg-ulb/creditcardfraud")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
df_raw, X, y, scaler = load_and_preprocess(uploaded)

with st.spinner("Training 4 models — ~60 seconds..."):
    results, X_test, y_test, best_thresh, feat_imp, X_train, y_train = train_models(X, y)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔴 FRAUD DETECTION SYSTEM")
fraud_count  = int(df_raw['Class'].sum())
total        = len(df_raw)
fraud_rate   = fraud_count / total
best_model   = max(results, key=lambda k: results[k]['auc'])
best_auc     = results[best_model]['auc']

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Cases",        f"{fraud_count:,}")
c3.metric("Fraud Rate",         f"{fraud_rate:.4%}")
c4.metric("Best ROC-AUC",       f"{best_auc:.4f}", delta=best_model)
c5.metric("Best Threshold",     f"{best_thresh:.2f}", delta="XGBoost tuned")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "🏆 Model Performance",
    "📈 Precision-Recall",
    "🔍 Feature Importance",
    "🎯 Predict Transaction"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Exploratory Data Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df_raw['Class'].value_counts()
        bars = ax.bar(['Normal', 'Fraud'], counts.values,
                      color=[BLUE, RED], edgecolor='none', width=0.45)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + counts.max() * 0.01,
                    f"{val:,}", ha='center', fontsize=10, color='#c8cdd6')
        ax.set_title("Class distribution", pad=10)
        ax.set_ylabel("Transactions")
        ax.grid(axis='y')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_raw[df_raw['Class'] == 0]['Amount'].clip(upper=500).hist(
            bins=60, ax=ax, alpha=0.6, color=BLUE, density=True, label='Normal')
        df_raw[df_raw['Class'] == 1]['Amount'].clip(upper=500).hist(
            bins=60, ax=ax, alpha=0.8, color=RED, density=True, label='Fraud')
        ax.set_title("Amount distribution (clipped at $500)", pad=10)
        ax.set_xlabel("Amount ($)")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    # Time distribution
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_raw[df_raw['Class'] == 0]['Time'].plot(
            kind='kde', ax=ax, color=BLUE, label='Normal', linewidth=2)
        df_raw[df_raw['Class'] == 1]['Time'].plot(
            kind='kde', ax=ax, color=RED, label='Fraud', linewidth=2)
        ax.set_title("Transaction time distribution", pad=10)
        ax.set_xlabel("Time (seconds from first transaction)")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_d:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        corr = df_raw.corr()['Class'].drop('Class').sort_values()
        top = pd.concat([corr.head(8), corr.tail(8)])
        colors = [RED if v < 0 else GREEN for v in top.values]
        top.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
        ax.set_title("Top feature correlations with fraud", pad=10)
        ax.set_xlabel("Correlation")
        ax.axvline(0, color='#5a6475', linewidth=0.8)
        ax.grid(axis='x')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    # Fraud amount stats
    st.markdown("#### Fraud vs normal — amount statistics")
    stats = df_raw.groupby('Class')['Amount'].describe().round(2)
    stats.index = ['Normal', 'Fraud']
    st.dataframe(stats, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance")

    # AUC summary
    cols = st.columns(4)
    colors_map = {'Logistic Regression': BLUE, 'Random Forest': GREEN,
                  'XGBoost': AMBER, 'Isolation Forest': RED}
    for i, (name, res) in enumerate(results.items()):
        cols[i].metric(name, f"AUC {res['auc']:.4f}",
                       delta="Best ✓" if name == best_model else None)

    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['probs'])
            ax.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})",
                    color=colors_map[name], linewidth=2)
        ax.plot([0,1],[0,1], '--', color='#1e2535', linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves — all models")
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_r:
        sel = st.selectbox("Confusion matrix for model", list(results.keys()))
        cm = confusion_matrix(y_test, results[sel]['preds'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal','Fraud'],
                    yticklabels=['Normal','Fraud'],
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix — {sel}")
        st.pyplot(fig); plt.close()

    # Classification reports
    st.markdown("#### Classification reports")
    rcols = st.columns(3)
    for i, (name, res) in enumerate(list(results.items())[:3]):
        rep = classification_report(y_test, res['preds'],
                                    target_names=['Normal','Fraud'],
                                    output_dict=True)
        with rcols[i]:
            st.markdown(f"**{name}**")
            rdf = pd.DataFrame(rep).T.loc[
                ['Normal','Fraud','macro avg'],
                ['precision','recall','f1-score','support']]
            st.dataframe(rdf.round(3), use_container_width=True)

    # Threshold tuning chart (XGBoost)
    st.markdown("#### Threshold tuning — XGBoost")
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, (results['XGBoost']['probs'] >= t).astype(int))
           for t in thresholds]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(thresholds, f1s, color=AMBER, linewidth=2)
    ax.axvline(best_thresh, linestyle='--', color=RED,
               label=f'Best threshold = {best_thresh:.2f}')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score (Fraud class)")
    ax.set_title("Threshold vs F1-Score — XGBoost")
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    st.pyplot(fig); plt.close()

    st.info(f"Optimal threshold = **{best_thresh:.2f}** — "
            f"transactions with fraud probability ≥ {best_thresh:.2f} are flagged as fraud.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Precision-Recall
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Precision-Recall Analysis")
    st.info("For fraud detection, **Precision-Recall** matters more than ROC-AUC "
            "because the dataset is extremely imbalanced (0.17% fraud). "
            "A high ROC-AUC can be misleading — always check the PR curve.")

    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, res in list(results.items())[:3]:
            prec, rec, _ = precision_recall_curve(y_test, res['probs'])
            ap = res['ap']
            ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})",
                    color=colors_map[name], linewidth=2)
        baseline = y_test.mean()
        ax.axhline(baseline, linestyle='--', color='#1e2535',
                   linewidth=1, label=f'Baseline ({baseline:.4f})')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curves")
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_r:
        # Precision vs Recall tradeoff for best model
        prec, rec, thresh = precision_recall_curve(
            y_test, results['XGBoost']['probs'])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(thresh, prec[:-1], color=GREEN, linewidth=2, label='Precision')
        ax.plot(thresh, rec[:-1],  color=RED,   linewidth=2, label='Recall')
        ax.axvline(best_thresh, linestyle='--', color=AMBER,
                   linewidth=1.5, label=f'Best threshold ({best_thresh:.2f})')
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.set_title("XGBoost — precision vs recall tradeoff")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        st.pyplot(fig); plt.close()

    # AP summary
    st.markdown("#### Average Precision (AP) scores")
    ap_data = {name: res['ap'] for name, res in results.items() if name != 'Isolation Forest'}
    ap_df = pd.DataFrame.from_dict(ap_data, orient='index', columns=['Average Precision'])
    ap_df['ROC-AUC'] = [results[n]['auc'] for n in ap_df.index]
    ap_df = ap_df.round(4).sort_values('Average Precision', ascending=False)
    st.dataframe(ap_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Feature Importance — XGBoost")
    top_n = st.slider("Show top N features", 5, 30, 15)
    fi = feat_imp.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, top_n * 0.45 + 1.5))
    bar_colors = [RED if i >= len(fi) - 3 else MUTED for i in range(len(fi))]
    bars = ax.barh(fi.index, fi.values, color=bar_colors, edgecolor='none', height=0.65)
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', fontsize=9, color='#c8cdd6')
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top {top_n} features driving fraud prediction")
    ax.grid(axis='x')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    top3 = feat_imp.head(3).index.tolist()
    st.info(f"**Top 3 fraud signals:** `{top3[0]}`, `{top3[1]}`, `{top3[2]}`  \n"
            f"V1–V28 are PCA-transformed features from the original transaction data "
            f"(anonymised for privacy). High importance on specific V-features indicates "
            f"those original dimensions are strong fraud signals.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Predict Transaction
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Predict Individual Transaction")
    st.caption("Enter transaction details to get a real-time fraud probability score.")

    pred_model_name = st.selectbox("Model to use for prediction",
        ['XGBoost', 'Random Forest', 'Logistic Regression'])

    use_thresh = st.checkbox(
        f"Use tuned threshold ({best_thresh:.2f}) instead of default 0.5",
        value=True)

    st.markdown("#### Transaction features")
    st.caption("V1–V28 are PCA components. Use values from -5 to 5. "
               "Amount and Time are in original scale.")

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0,
                                     max_value=25000.0, value=150.0, step=0.01)
            time   = st.number_input("Time (seconds from start)",
                                     min_value=0, max_value=172800, value=50000)
            st.markdown("**V1 – V14**")
            v_vals = []
            for i in range(1, 15):
                v_vals.append(st.number_input(f"V{i}", value=0.0,
                    min_value=-20.0, max_value=20.0, step=0.1, key=f"v{i}"))

        with col2:
            st.markdown("**V15 – V28**")
            for i in range(15, 29):
                v_vals.append(st.number_input(f"V{i}", value=0.0,
                    min_value=-20.0, max_value=20.0, step=0.1, key=f"v{i}"))

        submitted = st.form_submit_button("🔴 ANALYSE TRANSACTION")

    if submitted:
        # Build feature vector in correct column order
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        raw_vals = v_vals + [amount, time]
        input_df = pd.DataFrame([raw_vals], columns=feature_names)

        # Scale Amount and Time same way as training
        input_df['Amount'] = scaler.transform([[amount, 0]])[0][0]
        input_df['Time']   = scaler.transform([[0, time]])[0][1]

        # Reorder to match training column order
        input_df = input_df[X.columns]

        mdl = results[pred_model_name]['model']
        prob = mdl.predict_proba(input_df)[0, 1]
        if pred_model_name == 'Isolation Forest':
            score = -mdl.score_samples(input_df)[0]
            prob  = float(np.clip((score - 0.1) / 0.4, 0, 1))

        threshold = best_thresh if use_thresh else 0.5
        is_fraud  = prob >= threshold

        if is_fraud:
            st.markdown(f"""
            <div class="fraud-box fraud-alert">
                <div class="alert-badge">FRAUD ALERT</div>
                <div style="font-size:20px;font-weight:600;margin-bottom:6px">
                    Transaction flagged as FRAUDULENT
                </div>
                <div style="font-size:14px">Fraud probability: <strong>{prob:.2%}</strong></div>
                <div style="font-size:12px;margin-top:6px;opacity:0.7">
                    Model: {pred_model_name} · Threshold: {threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fraud-box fraud-safe">
                <div style="font-size:20px;font-weight:600;margin-bottom:6px">
                    Transaction appears LEGITIMATE
                </div>
                <div style="font-size:14px">Fraud probability: <strong>{prob:.2%}</strong></div>
                <div style="font-size:12px;margin-top:6px;opacity:0.7">
                    Model: {pred_model_name} · Threshold: {threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge bar
        fig, ax = plt.subplots(figsize=(7, 0.7))
        ax.barh([0], [1], color="#1e2535", height=0.5)
        ax.barh([0], [prob], color=RED if is_fraud else GREEN, height=0.5)
        ax.axvline(threshold, color=AMBER, linewidth=1.5,
                   linestyle='--', label=f'Threshold {threshold:.2f}')
        ax.set_xlim(0, 1)
        ax.axis('off')
        ax.set_facecolor("#080b10")
        fig.patch.set_facecolor("#080b10")
        ax.text(min(prob + 0.02, 0.95), 0,
                f"{prob:.1%}", va='center', color='#c8cdd6', fontsize=11)
        st.pyplot(fig); plt.close()

        # Risk breakdown
        st.markdown("#### Risk assessment")
        risk_level = "CRITICAL" if prob > 0.8 else \
                     "HIGH"     if prob > 0.5 else \
                     "MEDIUM"   if prob > 0.2 else "LOW"
        risk_color = RED if prob > 0.5 else AMBER if prob > 0.2 else GREEN

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud probability", f"{prob:.2%}")
        r2.metric("Risk level", risk_level)
        r3.metric("Decision", "BLOCK" if is_fraud else "ALLOW")