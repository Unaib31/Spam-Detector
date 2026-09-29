import streamlit as st

# PAGE CONFIG — MUST be the very first Streamlit command
st.set_page_config(
    page_title="SMS Spam Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import os
import re
import tensorflow as tf
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# SESSION STATE INITIALISATION
for key, default in [
    ("user_input_val", ""),
    ("_example_pending", None),   # staging key — never owned by a widget
    ("last_input", None),
    ("last_pred", None),
    ("last_confidence", None),
    ("last_nlp", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Promote any pending example text into the widget key BEFORE the
# text_area widget is instantiated.  Writing to a widget-owned key
# after the widget renders raises StreamlitWidgetAlreadyInstantiatedError.
if st.session_state["_example_pending"] is not None:
    st.session_state["user_input_val"] = st.session_state["_example_pending"]
    st.session_state["_example_pending"] = None

@st.cache_resource(show_spinner="Loading model assets...")
def load_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    model = tf.keras.models.load_model("spam_classifier.keras")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_resources()
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def analyse_text(raw: str) -> dict:
    lowered = raw.lower()
    cleaned = re.sub(r"[^a-z\s]", "", lowered)
    all_tokens = cleaned.split()
    no_stop = [w for w in all_tokens if w not in stop_words]
    removed_stopwords = [w for w in all_tokens if w in stop_words]
    stemmed_tokens = [ps.stem(w) for w in no_stop]
    preprocessed_text = " ".join(stemmed_tokens)
    vector = tfidf.transform([preprocessed_text]).toarray()
    raw_pred = float(model.predict(vector, verbose=0)[0][0])
    return {
        "raw": raw,
        "lowered": lowered,
        "cleaned": cleaned,
        "all_tokens": all_tokens,
        "no_stop": no_stop,
        "removed_stopwords": removed_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "preprocessed_text": preprocessed_text,
        "pred": raw_pred,
        "confidence": raw_pred * 100,
        "is_spam": raw_pred > 0.5,
        "char_count": len(raw),
        "word_count": len(raw.split()),
        "token_count": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "stopwords_removed": len(removed_stopwords),
    }

# RESTRAINED PROFESSIONAL LIGHT THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    background-color: #ffffff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1060px;
}

/* Header */
.app-header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.85rem;
    margin-bottom: 1.25rem;
}
.app-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 0.825rem;
    color: #64748b;
    margin-top: 0.2rem;
    margin-bottom: 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 6px 14px;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #0f172a !important;
    border-bottom: 2px solid #0f172a !important;
    font-weight: 600 !important;
}

/* Inputs & Buttons */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    font-size: 0.875rem !important;
    line-height: 1.5 !important;
    padding: 0.65rem !important;
}
.stTextArea textarea:focus {
    border-color: #0f172a !important;
    box-shadow: 0 0 0 1px #0f172a !important;
}
.stButton > button {
    border-radius: 4px;
    font-weight: 500;
    font-size: 0.825rem;
    border: 1px solid #cbd5e1;
    background-color: #ffffff;
    color: #334155;
    padding: 0.35rem 0.85rem;
    transition: all 0.1s ease;
}
.stButton > button:hover {
    border-color: #94a3b8;
    background-color: #f8fafc;
    color: #0f172a;
}
.stButton > button[kind="primary"] {
    background-color: #0f172a !important;
    border-color: #0f172a !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

/* Section Headings */
.ui-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569;
    margin-bottom: 0.4rem;
}

/* Output Containers */
.ui-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.ui-panel-muted {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

/* Status Cards */
.result-status-spam {
    background-color: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 4px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.result-status-ham {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 4px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.result-title-spam {
    font-size: 0.9rem;
    font-weight: 700;
    color: #9f1239;
    letter-spacing: 0.02em;
}
.result-title-ham {
    font-size: 0.9rem;
    font-weight: 700;
    color: #166534;
    letter-spacing: 0.02em;
}

/* Data summary chips */
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.5rem;
    text-align: center;
}
.stat-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
}
.stat-lbl {
    font-size: 0.68rem;
    color: #64748b;
    margin-top: 1px;
}

/* Token representations */
.token-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 1px 5px;
    margin: 2px;
}
.token-chip-removed {
    background: #fef2f2;
    color: #991b1b;
    border-color: #fee2e2;
    text-decoration: line-through;
}

/* Pipeline Flow Diagram */
.flow-container {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 6px;
    padding: 0.4rem 0;
    overflow-x: auto;
}
.flow-card {
    flex: 1;
    min-width: 145px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.6rem 0.7rem;
}
.flow-badge {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.03em;
    margin-bottom: 2px;
}
.flow-text-main {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
}
.flow-text-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 2px;
    line-height: 1.3;
}
.flow-arrow {
    display: flex;
    align-items: center;
    color: #94a3b8;
    font-weight: 700;
    font-size: 0.95rem;
    user-select: none;
    padding: 0 2px;
}

/* Technical tables */
.tech-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.tech-table th {
    text-align: left;
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid #cbd5e1;
    color: #475569;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
}
.tech-table td {
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
}
.tech-table tr:last-child td {
    border-bottom: none;
}
.code-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 2px;
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="app-header">
  <h1 class="app-title">SMS Spam Classifier</h1>
  <p class="app-subtitle">NLP inference engine and feature extraction pipeline</p>
</div>
""", unsafe_allow_html=True)

tab_detector, tab_nlp, tab_model = st.tabs(["Detector", "NLP Pipeline", "Model"])

# TAB 1: DETECTOR
with tab_detector:
    # PIPELINE OVERVIEW TOGGLE
    with st.expander("Pipeline Overview · How the Model Works", expanded=False):
        st.markdown(r"""
        <div class="flow-container">
          <div class="flow-card">
            <div class="flow-badge">1. Input</div>
            <div class="flow-text-main">Raw SMS</div>
            <div class="flow-text-sub">Unstructured text input</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">
            <div class="flow-badge">2. Preprocessing</div>
            <div class="flow-text-main">NLP Cleaning</div>
            <div class="flow-text-sub">Lowercase, regex, stopword filter & Porter stemmer</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">
            <div class="flow-badge">3. Vectorisation</div>
            <div class="flow-text-main">TF-IDF</div>
            <div class="flow-text-sub">Sparse numeric representation (3,000 features)</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">
            <div class="flow-badge">4. Classifier</div>
            <div class="flow-text-main">Neural Network</div>
            <div class="flow-text-sub">Dense(128) → Dropout(0.5) → Dense(64) → Sigmoid</div>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">
            <div class="flow-badge">5. Decision</div>
            <div class="flow-text-main">Threshold (0.50)</div>
            <div class="flow-text-sub">$\ge 0.50 \implies$ <b>SPAM</b>, else <b>HAM</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="ui-label">Message Input</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            label="SMS text",
            label_visibility="collapsed",
            placeholder="Enter an SMS message to evaluate...",
            height=130,
            key="user_input_val",
        )

        b1, b2 = st.columns([2, 1])
        analyze_clicked = b1.button("Classify Message", use_container_width=True, type="primary")
        clear_clicked   = b2.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state["_example_pending"] = ""  # promoted on rerun
            st.session_state["last_input"] = None
            st.session_state["last_pred"] = None
            st.session_state["last_confidence"] = None
            st.session_state["last_nlp"] = None
            st.rerun()

        if analyze_clicked:
            raw = user_input.strip()
            if not raw:
                st.warning("Please enter a message to classify.")
            else:
                with st.spinner("Classifying..."):
                    result = analyse_text(raw)
                st.session_state["last_input"]      = raw
                st.session_state["last_pred"]       = result["pred"]
                st.session_state["last_confidence"] = result["confidence"]
                st.session_state["last_nlp"]        = result

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ui-label">Sample Inputs</div>', unsafe_allow_html=True)
        examples = [
            ("Sample Ham", "Hey! Are we still on for lunch tomorrow? Let me know."),
            ("Sample Spam", "CONGRATULATIONS! You won a $1,000 Gift Card. Claim now: bit.ly/win1000"),
            ("Sample Phishing", "Your package is on hold. Confirm your address: http://parcel-track.biz/claim"),
        ]
        ec1, ec2, ec3 = st.columns(3)
        for col, (label, msg) in zip([ec1, ec2, ec3], examples):
            if col.button(label, use_container_width=True):
                st.session_state["_example_pending"] = msg  # promoted on rerun
                st.rerun()

    with right:
        st.markdown('<div class="ui-label">Prediction Result</div>', unsafe_allow_html=True)

        if st.session_state["last_pred"] is None:
            st.markdown("""
            <div class="ui-panel-muted" style="text-align:center; padding: 2.2rem 1rem;">
              <p style="color:#64748b; margin:0; font-size:0.85rem;">
                Enter a message and click <b>Classify Message</b> to inspect classification output.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            pred       = st.session_state["last_pred"]
            confidence = st.session_state["last_confidence"]
            is_spam    = pred > 0.5
            display_conf = confidence if is_spam else (100 - confidence)

            if is_spam:
                st.markdown(f"""
                <div class="result-status-spam">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-title-spam">SPAM</span>
                    <span style="font-size:0.8rem; color:#475569;">Confidence: <b>{confidence:.1f}%</b></span>
                  </div>
                  <div style="font-size:0.8rem; color:#475569; margin-top:0.25rem;">Classified as unsolicited/spam.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-status-ham">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-title-ham">HAM (LEGITIMATE)</span>
                    <span style="font-size:0.8rem; color:#475569;">Confidence: <b>{100 - confidence:.1f}%</b></span>
                  </div>
                  <div style="font-size:0.8rem; color:#475569; margin-top:0.25rem;">Classified as legitimate communication.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.75rem; color:#64748b; margin: 0.4rem 0 0.2rem 0;">Probability Score</div>', unsafe_allow_html=True)
            st.progress(int(display_conf), text=f"{display_conf:.1f}%")

            nlp_quick = st.session_state["last_nlp"]
            if nlp_quick:
                st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="stat-box"><div class="stat-val">{nlp_quick["word_count"]}</div><div class="stat-lbl">Words</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="stat-box"><div class="stat-val">{nlp_quick["token_count"]}</div><div class="stat-lbl">Tokens</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="stat-box"><div class="stat-val">{nlp_quick["stopwords_removed"]}</div><div class="stat-lbl">Stopwords Removed</div></div>', unsafe_allow_html=True)


# TAB 2: NLP PIPELINE
with tab_nlp:
    nlp = st.session_state.get("last_nlp")

    if nlp is None:
        st.info("Evaluate a message in the Detector tab to view pipeline processing data.")
    else:
        st.markdown('<div class="ui-label">Transformation Pipeline Stages</div>', unsafe_allow_html=True)
        pipeline_steps = [
            ("1. Input String",              nlp["raw"]),
            ("2. Lowercase Normalisation",   nlp["lowered"]),
            ("3. Non-Alpha Filtering",       nlp["cleaned"]),
            ("4. Stopword Removal",          " ".join(nlp["no_stop"]) if nlp["no_stop"] else "(empty)"),
            ("5. Porter Stemming",           nlp["preprocessed_text"] if nlp["preprocessed_text"] else "(empty)"),
            ("6. TF-IDF Transformation",      f"Sparse Vector (1, {tfidf.get_feature_names_out().shape[0]:,})"),
            ("7. Feed-Forward Neural Net",    f"Sigmoid Output: {nlp['pred']:.4f}"),
            ("8. Binary Threshold (0.50)",   f"{'SPAM' if nlp['is_spam'] else 'HAM'} ({nlp['confidence']:.1f}% spam score)"),
        ]

        table_rows = "".join(
            f'<tr><td style="font-weight:600; width:220px;">{step}</td><td><span class="code-val">{val}</span></td></tr>'
            for step, val in pipeline_steps
        )
        st.markdown(f"""
        <div class="ui-panel" style="padding:0.25rem 0.6rem;">
          <table class="tech-table">
            <thead>
              <tr><th>Pipeline Stage</th><th>Intermediate Output</th></tr>
            </thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="ui-label">Raw Text</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ui-panel-muted" style="font-size:0.85rem; color:#334155;">{nlp["raw"]}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="ui-label">Preprocessed String</div>', unsafe_allow_html=True)
            processed_display = nlp["preprocessed_text"] if nlp["preprocessed_text"] else "(empty)"
            st.markdown(f'<div class="ui-panel-muted"><span class="code-val">{processed_display}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="ui-label">Text Statistics</div>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, val, label in [
            (s1, nlp["char_count"],       "Characters"),
            (s2, nlp["word_count"],       "Words"),
            (s3, nlp["token_count"],      "Tokens"),
            (s4, nlp["unique_tokens"],    "Unique Tokens"),
            (s5, nlp["stopwords_removed"],"Stopwords Filtered"),
        ]:
            col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lbl">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)
        col_tok1, col_tok2 = st.columns(2)
        with col_tok1:
            st.markdown('<div class="ui-label">Retained Tokens</div>', unsafe_allow_html=True)
            if nlp["no_stop"]:
                pills = " ".join(f'<span class="token-chip">{t}</span>' for t in nlp["no_stop"])
                st.markdown(f'<div class="ui-panel">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ui-panel-muted" style="font-size:0.8rem; color:#64748b;">No tokens retained.</div>', unsafe_allow_html=True)

        with col_tok2:
            st.markdown('<div class="ui-label">Filtered Stopwords</div>', unsafe_allow_html=True)
            if nlp["removed_stopwords"]:
                pills = " ".join(f'<span class="token-chip token-chip-removed">{t}</span>' for t in nlp["removed_stopwords"])
                st.markdown(f'<div class="ui-panel">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ui-panel-muted" style="font-size:0.8rem; color:#64748b;">No stopwords filtered.</div>', unsafe_allow_html=True)


# TAB 3: MODEL
with tab_model:
    col_info, col_arch = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown('<div class="ui-label">Dataset Specification</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-panel">
          <table class="tech-table">
            <tr><td>Source Dataset</td><td style="font-weight:600; text-align:right;">UCI SMS Spam Collection</td></tr>
            <tr><td>Total Messages</td><td style="text-align:right;">5,574</td></tr>
            <tr><td>Legitimate (Ham)</td><td style="text-align:right;">4,827 (86.6%)</td></tr>
            <tr><td>Spam</td><td style="text-align:right;">747 (13.4%)</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ui-label">TF-IDF Vectorizer</div>', unsafe_allow_html=True)
        try:
            n_features = tfidf.get_feature_names_out().shape[0]
        except Exception:
            n_features = 3000
        st.markdown(f"""
        <div class="ui-panel">
          <table class="tech-table">
            <tr><td>Max Features (Vocabulary)</td><td style="font-weight:600; text-align:right;">{n_features:,}</td></tr>
            <tr><td>Output Vector Shape</td><td style="text-align:right;"><span class="code-val">(1, {n_features})</span></td></tr>
            <tr><td>Weighting Method</td><td style="text-align:right;">Term Frequency – Inverse Document Frequency</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ui-label">Performance Metrics Notice</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-panel-muted" style="font-size:0.8rem; color:#64748b; line-height:1.4;">
          Evaluation metrics (accuracy, precision, recall, F1) depend on the test split used during evaluation and are not stored inside serialized model files. Run an evaluation script against the test split to generate verified metrics.
        </div>
        """, unsafe_allow_html=True)

    with col_arch:
        st.markdown('<div class="ui-label">Model Architecture</div>', unsafe_allow_html=True)
        # Verified parameters from spam_classifier.keras
        st.markdown("""
        <div class="ui-panel" style="padding:0.25rem 0.6rem;">
          <table class="tech-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Output Shape</th>
                <th>Activation</th>
                <th style="text-align:right;">Params</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>Dense</b> (128 units)</td>
                <td><span class="code-val">(None, 128)</span></td>
                <td>ReLU</td>
                <td style="text-align:right; font-family:monospace;">384,128</td>
              </tr>
              <tr>
                <td><b>Dropout</b> (0.50)</td>
                <td><span class="code-val">(None, 128)</span></td>
                <td>—</td>
                <td style="text-align:right; font-family:monospace;">0</td>
              </tr>
              <tr>
                <td><b>Dense</b> (64 units)</td>
                <td><span class="code-val">(None, 64)</span></td>
                <td>ReLU</td>
                <td style="text-align:right; font-family:monospace;">8,256</td>
              </tr>
              <tr>
                <td><b>Dense</b> (1 unit)</td>
                <td><span class="code-val">(None, 1)</span></td>
                <td>Sigmoid</td>
                <td style="text-align:right; font-family:monospace;">65</td>
              </tr>
              <tr style="border-top:1px solid #cbd5e1; font-weight:600;">
                <td colspan="3">Total Trainable Parameters</td>
                <td style="text-align:right; font-family:monospace; color:#0f172a;">392,449</td>
              </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ui-label">Classification Configuration</div>', unsafe_allow_html=True)
        st.markdown(r"""
        <div class="ui-panel">
          <table class="tech-table">
            <tr><td>Model Type</td><td style="font-weight:600; text-align:right;">Keras Sequential</td></tr>
            <tr><td>Decision Threshold</td><td style="text-align:right;">0.50</td></tr>
            <tr><td>Decision Rule</td><td style="text-align:right;">$\hat{y} \ge 0.50 \implies$ <b>SPAM</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="border-top: 1px solid #e2e8f0; margin-top: 2rem; padding-top: 0.75rem; text-align: center; font-size: 0.75rem; color: #94a3b8;">
  SMS Spam Classifier · NLP Case Study
</div>
""", unsafe_allow_html=True)
