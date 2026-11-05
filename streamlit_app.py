import html
import streamlit as st
from transformers import pipeline

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Named Entity Recognition - CoNLL-2003",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "Abd3laziz1/ner-distilbert-conll2003"
MAX_CHARS = 1500

# ---------------- Load HF Model ----------------
@st.cache_resource(show_spinner="⚡ Loading DistilBERT Model...")
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)

ner = load_model()

# ---------------- Exact HTML CSS & Glassmorphism Theme ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Dark Setup */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Title */
    .hero-title {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-top: -20px;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 25px;
        font-weight: 500;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* HTML HTML Token Highlights (Exact HTML match) */
    .token-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        margin: 3px 2px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.4;
        border: 1px solid rgba(255, 255, 255, 0.12);
        transition: transform 0.2s ease;
    }
    
    .token-badge:hover {
        transform: translateY(-2px);
    }

    .tag-PER  { background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.35); }
    .tag-ORG  { background: rgba(236, 72, 153, 0.18); color: #f472b6; border-color: rgba(236, 72, 153, 0.35); }
    .tag-LOC  { background: rgba(34, 197, 94, 0.18);  color: #4ade80; border-color: rgba(34, 197, 94, 0.35); }
    .tag-MISC { background: rgba(234, 179, 8, 0.18);  color: #facc15; border-color: rgba(234, 179, 8, 0.35); }

    .tag-label {
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        padding: 2px 5px;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.4);
        letter-spacing: 0.5px;
    }

    /* Input Controls Customization */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.25) !important;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "German chancellor Angela Merkel met French president Emmanuel Macron in Brussels to discuss European Union affairs."
if "submitted" not in st.session_state:
    st.session_state["submitted"] = True

def clear_text():
    st.session_state["user_text"] = ""
    st.session_state["submitted"] = False

# ---------------- UI Header ----------------
st.markdown('<div class="hero-title">Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Extract Person, Organization, Location & Miscellaneous entities using DistilBERT CoNLL-2003</div>', unsafe_allow_html=True)

# ---------------- Top Cards Layout (Legend & Input) ----------------
col_left, col_right = st.columns([1, 2.2], gap="medium")

with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🏷️ Entity Legend</div>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
            <div class="token-badge tag-PER"><span style="color:#f8fafc">Person</span> <span class="tag-label">PER</span></div>
            <div class="token-badge tag-ORG"><span style="color:#f8fafc">Organization</span> <span class="tag-label">ORG</span></div>
            <div class="token-badge tag-LOC"><span style="color:#f8fafc">Location</span> <span class="tag-label">LOC</span></div>
            <div class="token-badge tag-MISC"><span style="color:#f8fafc">Miscellaneous</span> <span class="tag-label">MISC</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card-title">✍️ Write your own</div>', unsafe_allow_html=True)
    
    input_text = st.text_area(
        label="Input Text",
        value=st.session_state["user_text"],
        height=130,
        placeholder="Type or paste your text here...",
        label_visibility="collapsed"
    )

    btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 5])
    with btn_col1:
        submit_btn = st.button("🚀 Submit", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("🗑️ Clear", on_click=clear_text, type="secondary", use_container_width=True)

    if submit_btn:
        st.session_state["user_text"] = input_text
        st.session_state["submitted"] = True

# ---------------- Prediction & Output Section ----------------
st.markdown("---")
st.markdown('<div class="card-title">📊 Visual Extraction & Entities</div>', unsafe_allow_html=True)

current_text = input_text.strip()[:MAX_CHARS]

if st.session_state["submitted"] and current_text:
    # Inference call
    entities = sorted(ner(current_text), key=lambda e: e["start"])

    parts, rows, pos = [], [], 0
    for e in entities:
        start, end = max(e["start"], pos), e["end"]
        if start >= end:
            continue
        label = e["entity_group"]
        
        # Plain text segment
        parts.append(html.escape(current_text[pos:start]))
        
        # Styled HTML Tag Badge (identical to HTML design)
        parts.append(
            f'<span class="token-badge tag-{label}">'
            f'<span>{html.escape(current_text[start:end])}</span>'
            f'<span class="tag-label">{label}</span>'
            f'</span>'
        )
        
        rows.append({
            "Entity Text": current_text[start:end],
            "Category": label,
            "Confidence Score": f"{float(e['score']):.2%}"
        })
        pos = end

    parts.append(html.escape(current_text[pos:]))

    # Display Text Visualization Inside a Glass Container
    st.markdown(
        f'<div class="glass-card" style="line-height: 2.3; font-size: 1.05rem;">'
        + "".join(parts) +
        '</div>',
        unsafe_allow_html=True
    )
    
    # Display Results Table
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No entities detected in the text.")
else:
    st.info("Enter your text above and click **Submit** to run prediction.")
