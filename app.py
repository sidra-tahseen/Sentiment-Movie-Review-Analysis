"""
Wheeltrix Project 1 — Movie Review Sentiment Analysis demo.

Compares a classical TF-IDF+LogReg baseline against a fine-tuned DistilBERT,
shows confidence for both, and explains predictions with LIME word highlights.
Visual design follows the "Wheeltrix Sentiment AI Lab" dark theme.

Local run:
    pip install -r requirements.txt
    streamlit run app.py

Deploy free at https://share.streamlit.io (Streamlit Community Cloud) by
pointing it at this repo/folder — no server setup needed.
"""
import html
import os
import re
import time
import joblib
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Wheeltrix Sentiment AI Lab", page_icon="🎬", layout="centered")

# Prefer the negation-aware baseline (config D from the Colab experiments) if its file
# is present; otherwise fall back to the original baseline so the app still runs.
NEG_PATH = "logreg_negation_pipeline.joblib"
USE_NEG_MARKING = os.path.exists(NEG_PATH)
LOGREG_PATH = NEG_PATH if USE_NEG_MARKING else "logreg_pipeline.joblib"
# Use the local model folder if it exists (your laptop); otherwise download from the Hugging Face Hub (deployed).
HUB_MODEL_ID = "sidratahseen/distilbert-imdb-sentiment"
BERT_PATH = "distilbert-imdb-final" if os.path.isdir("distilbert-imdb-final") else HUB_MODEL_ID

# ---- palette (from the Wheeltrix design system) ----
CANVAS = "#0b0f19"
SURFACE = "#0f172a"
SURFACE_EL = "#171f33"
SURFACE_HI = "#222a3d"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#dae2fd"
TEXT_MUTE = "#9aa3c0"
PRIMARY = "#8083ff"        # indigo
SECONDARY = "#a78bfa"      # violet
POS = "#10b981"            # emerald
NEG = "#f43f5e"            # rose
AMBER = "#f59e0b"

# Below this confidence a model's answer is shown as "UNCERTAIN" instead of a label.
# A design choice, not a calibrated value: tune it if you like.
UNSURE_BELOW = 0.60

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {CANVAS}; }}
    header[data-testid="stHeader"] {{ background: {CANVAS} !important; }}
    .block-container {{ max-width: 860px; padding-top: 3.2rem !important; }}
    h1, h2, h3 {{ color: {TEXT} !important; }}
    p, span, div, label {{ color: {TEXT}; }}
    code, .mono {{ font-family: 'JetBrains Mono', monospace !important; }}

    .eyebrow {{
        display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.7rem;
        border-radius:999px; background:rgba(99,102,241,0.18); color:{SECONDARY};
        font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;
    }}
    .hero-title {{
        font-size:2.1rem; font-weight:800; letter-spacing:-0.02em; line-height:1.2;
        background:linear-gradient(90deg,{TEXT},{PRIMARY} 70%,{SECONDARY});
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:0.5rem;
    }}
    .subtitle {{ color:{TEXT_MUTE}; font-size:1rem; line-height:1.6; margin-bottom:1.1rem; max-width:640px; }}

    .status-row {{ display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1.4rem; }}
    .status-chip {{
        display:inline-flex; align-items:center; gap:0.4rem; padding:0.4rem 0.75rem;
        border-radius:10px; background:{SURFACE_HI}; border:1px solid {BORDER};
        font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{TEXT};
    }}
    .status-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; box-shadow:0 0 6px currentColor; }}

    div[data-testid="stTextArea"] textarea {{
        background:{CANVAS} !important; color:{TEXT} !important;
        border:1px solid {BORDER} !important; border-radius:12px !important;
        font-family:'JetBrains Mono',monospace !important; font-size:1rem !important;
    }}
    div[data-baseweb="select"] > div {{
        background:{SURFACE_EL} !important; border-color:{BORDER} !important; color:{TEXT} !important;
        border-radius:10px !important;
    }}
    div[data-testid="stSelectbox"] label {{ color:{TEXT_MUTE} !important; font-size:0.85rem; }}

    .stButton > button, [data-testid="stFormSubmitButton"] > button {{
        background:linear-gradient(135deg,{PRIMARY},{SECONDARY}) !important; color:#07006c !important;
        border:none !important; border-radius:10px !important; font-weight:700 !important;
        box-shadow:0 0 20px rgba(128,131,255,0.35) !important; padding:0.55rem 1.4rem !important;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ box-shadow:0 0 28px rgba(128,131,255,0.55) !important; }}

    .model-card {{
        border:1px solid {BORDER}; border-radius:16px; padding:1.3rem 1.4rem;
        background:{SURFACE_EL}; box-shadow:0 4px 20px -4px rgba(0,0,0,0.5);
    }}
    .model-arch {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700;
                  text-transform:uppercase; letter-spacing:0.07em; color:{PRIMARY}; margin-bottom:0.2rem; }}
    .model-name {{ font-size:1.15rem; font-weight:700; color:{TEXT}; margin-bottom:0.9rem; }}
    .verdict-pill {{
        display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.8rem;
        border-radius:999px; font-weight:700; font-size:0.95rem; margin-bottom:0.7rem;
    }}
    .model-conf-label {{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{TEXT_MUTE}; }}
    .model-conf-val {{ font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; }}
    .bar-track {{ background:{SURFACE_HI}; border-radius:999px; height:10px; margin:0.5rem 0 0.3rem; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; transition:width 0.5s; }}
    .bar-labels {{ display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
                  font-size:0.65rem; color:{TEXT_MUTE}; text-transform:uppercase; letter-spacing:0.04em; }}

    .disagree-banner {{
        display:flex; gap:0.9rem; align-items:flex-start; padding:1rem 1.2rem; border-radius:14px;
        background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.35); margin:1.2rem 0;
    }}

    .word {{ padding:0.08rem 0.18rem; border-radius:5px; }}
    .legend-chip {{ display:inline-flex; align-items:center; gap:0.35rem; font-family:'JetBrains Mono',monospace;
                   font-size:0.75rem; color:{TEXT_MUTE}; margin-right:1.1rem; text-transform:uppercase; letter-spacing:0.04em; }}
    .legend-dot {{ width:9px; height:9px; border-radius:3px; display:inline-block; }}

    .attr-row {{ display:flex; align-items:center; gap:0.7rem; margin:0.35rem 0; }}
    .attr-word {{ width:110px; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:{TEXT}; flex-shrink:0; }}
    .attr-track {{ flex:1; background:{SURFACE_HI}; border-radius:6px; height:8px; overflow:hidden; }}
    .attr-fill {{ height:100%; border-radius:6px; }}
    .attr-val {{ width:56px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{TEXT_MUTE}; }}

    div[data-testid="stForm"] {{ border:none !important; padding:0 !important; }}
    div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:14px; background:{SURFACE}; }}
    div[data-testid="stExpander"] summary {{ color:{TEXT} !important; }}
    hr {{ border-color:{BORDER} !important; }}
    .stTabs [data-baseweb="tab"] {{ color:{TEXT_MUTE}; font-family:'JetBrains Mono',monospace; font-size:0.85rem; }}
    .stTabs [aria-selected="true"] {{ color:{PRIMARY} !important; }}
</style>
""", unsafe_allow_html=True)


# ---------- loading (cached so it only happens once per session) ----------
@st.cache_resource
def load_logreg():
    return joblib.load(LOGREG_PATH)


@st.cache_resource
def load_bert():
    tok = AutoTokenizer.from_pretrained(BERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    model.eval()
    return tok, model


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


_NEGATORS = {"not", "no", "never", "cannot", "without", "nothing", "neither", "nor"}
_PUNCT = set(".,!?;:")


def mark_negation(text: str, scope: int = 3) -> str:
    """Must match the function used when the negation-aware baseline was trained:
    'not good' -> 'not NOT_good', so negated words become their own features."""
    text = re.sub(r"n't\b", " not", text)
    toks = re.findall(r"[a-z']+|[.,!?;:]", text)
    out, remaining = [], 0
    for t in toks:
        if t in _PUNCT:
            remaining = 0
            out.append(t)
        elif t in _NEGATORS:
            out.append(t)
            remaining = scope
        elif remaining > 0:
            out.append("NOT_" + t)
            remaining -= 1
        else:
            out.append(t)
    return " ".join(out)


def lr_proba(texts, pipe):
    prep = (lambda t: mark_negation(clean(t))) if USE_NEG_MARKING else clean
    p = pipe.predict_proba([prep(t) for t in texts])[:, 1]
    return np.vstack([1 - p, p]).T


def bert_proba(texts, tok, model):
    enc = tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    p = torch.softmax(logits, -1)[:, 1].numpy()
    return np.vstack([1 - p, p]).T


def word_highlight_html(text: str, weights: dict) -> str:
    """Render `text` with each word's background tinted by its LIME weight."""
    max_abs = max((abs(v) for v in weights.values()), default=1.0) or 1.0
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]|\s+", text)
    out = []
    lower_weights = {k.lower(): v for k, v in weights.items()}
    for tok in tokens:
        key = tok.lower()
        if key in lower_weights and tok.strip():
            w = lower_weights[key]
            alpha = min(abs(w) / max_abs, 1.0) * 0.55 + 0.12
            rgb = "16,185,129" if w > 0 else "244,63,94"
            out.append(f'<span class="word" style="background:rgba({rgb},{alpha:.2f})" '
                      f'title="{w:+.2f}">{html.escape(tok)}</span>')
        else:
            out.append(html.escape(tok))
    return "".join(out)


def attribution_bars_html(weights: dict, n=6) -> str:
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:n]
    max_abs = max((abs(v) for _, v in top), default=1.0) or 1.0
    rows = []
    for w, v in top:
        color = POS if v > 0 else NEG
        width = abs(v) / max_abs * 100
        rows.append(f"""
        <div class="attr-row">
            <div class="attr-word">{html.escape(w)}</div>
            <div class="attr-track"><div class="attr-fill" style="width:{width:.0f}%; background:{color}"></div></div>
            <div class="attr-val">{v:+.2f}</div>
        </div>""")
    return "".join(rows)


# ---------- header ----------
st.markdown('<div class="eyebrow">🎬 &nbsp;LIME-Integrated Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Movie Review Sentiment Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classical TF-IDF + Logistic Regression vs. fine-tuned DistilBERT, '
    'trained on 25,000 IMDB movie reviews. Compare predictions and inspect token-level '
    'attributions in real time.</div>', unsafe_allow_html=True
)
st.markdown(f"""
<div class="status-row">
    <span class="status-chip"><span class="status-dot" style="background:{POS};color:{POS}"></span>LogReg: Ready · 25k IMDB</span>
    <span class="status-chip"><span class="status-dot" style="background:{PRIMARY};color:{PRIMARY}"></span>DistilBERT: Fine-tuned</span>
    <span class="status-chip"><span class="status-dot" style="background:{SECONDARY};color:{SECONDARY}"></span>LIME Explainer: Online</span>
</div>
""", unsafe_allow_html=True)

examples = {
    "-- pick an example --": "",
    "Straightforward positive": "A heartfelt story with superb performances from the entire cast.",
    "Negation (tricky)": "Not bad, not bad at all.",
    "Sarcasm (tricky)": "Fantastic, the acting is so wooden it could be used as furniture.",
    "Mixed review": "Great acting and a wonderful score, though the ending was a bit rushed.",
}
choice = st.selectbox("Test corpus — try an example, or write your own below:", list(examples.keys()))
default_text = examples[choice]
# The text box and button live in a form so the typed text and the click are sent
# together. (A plain disabled button swallowed the first click: the click made the box
# lose focus, which committed the text and re-rendered the button mid-click.)
with st.form("analyze_form"):
    text = st.text_area("Review text console:", value=default_text, height=120,
                        placeholder="Type or paste a movie review here...")
    run = st.form_submit_button("⚡ Run Analysis", type="primary")

if run and not text.strip():
    st.info("Type or paste a review first, or pick an example above.")

if run and text.strip():
    try:
        lr_pipe = load_logreg()
        tok, model = load_bert()
    except Exception as e:
        st.error(
            "Couldn't load the model files. Make sure `logreg_pipeline.joblib` and the "
            f"`{BERT_PATH}/` folder (from your Colab notebook) are in this app's directory.\n\n"
            f"Details: {e}"
        )
        st.stop()

    t0 = time.perf_counter()
    p_lr = lr_proba([text], lr_pipe)[0, 1]
    lr_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    p_bert = bert_proba([text], tok, model)[0, 1]
    bert_ms = (time.perf_counter() - t0) * 1000
    bert_device = "GPU" if torch.cuda.is_available() else "CPU"

    if abs(p_lr - p_bert) > 0.3:
        st.markdown(f"""
        <div class="disagree-banner">
            <div style="font-size:1.4rem;">⚠️</div>
            <div>
                <div style="font-weight:700; margin-bottom:0.3rem;">High Model Disagreement Detected
                    <span class="mono" style="background:{AMBER}; color:#1a1200; padding:0.15rem 0.5rem;
                    border-radius:6px; font-size:0.72rem; margin-left:0.5rem;">|Δp| = {abs(p_lr-p_bert):.2f} &gt; 0.30</span></div>
                <div style="color:{TEXT_MUTE}; font-size:0.9rem;">The two architectures reach opposing
                conclusions — often a sign of negation, sarcasm, or mixed sentiment. See the
                token attributions below.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    model_meta = {
        "Logistic Regression": ("Linear Baseline Architecture", "TF-IDF 1-2 Grams · Negation-Marked" if USE_NEG_MARKING else "TF-IDF 1-2 Grams · Bag-of-Words"),
        "DistilBERT": ("Transformer Architecture", "6-Layer Self-Attention · Fine-Tuned"),
    }
    cards = [
        (col1, "Logistic Regression", p_lr, lr_ms, "CPU"),
        (col2, "DistilBERT", p_bert, bert_ms, bert_device),
    ]
    for col, name, p, ms, device in cards:
        conf = p if p > 0.5 else 1 - p
        if conf < UNSURE_BELOW:
            label, color, pill_rgb = "UNCERTAIN", AMBER, "245,158,11"
        elif p > 0.5:
            label, color, pill_rgb = "POSITIVE", POS, "16,185,129"
        else:
            label, color, pill_rgb = "NEGATIVE", NEG, "244,63,94"
        arch, sub = model_meta[name]
        col.markdown(f"""
        <div class="model-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="model-arch">{arch}</div>
                <span class="mono" style="font-size:0.68rem; color:{TEXT_MUTE}; background:{SURFACE_HI};
                      padding:0.2rem 0.5rem; border-radius:6px; border:1px solid {BORDER}; white-space:nowrap;">
                      ⏱ {ms:.1f} ms {device}</span>
            </div>
            <div class="model-name">{name}<br><span style="font-size:0.75rem; font-weight:400; color:{TEXT_MUTE}; font-family:'JetBrains Mono',monospace;">{sub}</span></div>
            <div class="verdict-pill" style="background:rgba({pill_rgb},0.18); color:{color};">{label}</div>
            <div class="model-conf-label">Confidence</div>
            <div class="model-conf-val" style="color:{color}">{conf:.1%}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{p*100:.0f}%; background:{color}"></div></div>
            <div class="bar-labels"><span>Negative</span><span>Positive</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Why did each model decide that?")
    st.markdown(
        f'<span class="legend-chip"><span class="legend-dot" style="background:{POS}"></span>'
        f'pushed toward positive</span>'
        f'<span class="legend-chip"><span class="legend-dot" style="background:{NEG}"></span>'
        f'pushed toward negative</span>', unsafe_allow_html=True
    )
    st.write("")

    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    tabs = st.tabs(["Logistic Regression", "DistilBERT"])
    fns = {"Logistic Regression": lambda t: lr_proba(t, lr_pipe),
          "DistilBERT": lambda t: bert_proba(t, tok, model)}

    for tab, name in zip(tabs, fns):
        with tab:
            with st.spinner(f"Running LIME for {name}..."):
                exp = explainer.explain_instance(text, fns[name], num_features=10, num_samples=500)
            weights = dict(exp.as_list())
            st.markdown(
                f'<div class="model-card" style="font-size:1.05rem; line-height:2.1; '
                f'font-family:\'JetBrains Mono\',monospace;">'
                f'{word_highlight_html(text, weights)}</div>', unsafe_allow_html=True
            )
            st.write("")
            st.markdown(
                f'<div class="mono" style="font-size:0.7rem; color:{TEXT_MUTE}; '
                f'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.3rem;">'
                f'Ranked feature contributions</div>', unsafe_allow_html=True
            )
            st.markdown(attribution_bars_html(weights), unsafe_allow_html=True)

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
        Built for the **Wheeltrix Data Science & Analytics internship**.

        - Baseline: TF-IDF (1-2 grams) + Logistic Regression, trained on 25k IMDB reviews.
        - Transformer: DistilBERT fine-tuned for 2 epochs on the same data.
        - Both were stress-tested on negation, sarcasm, and cross-domain data
          (Rotten Tomatoes, SST-2) — DistilBERT generalizes better out-of-domain,
          but both models struggle with negation and sarcasm in different ways.
        - Explanations use [LIME](https://github.com/marcotcr/lime).
        """
    )