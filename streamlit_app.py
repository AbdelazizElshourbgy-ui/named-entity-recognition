import html
import streamlit as st
from transformers import pipeline

# ---------------- Page Config & Setup ----------------
st.set_page_config(
    page_title="NER Predictor - CoNLL-2003",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "Abd3laziz1/ner-distilbert-conll2003"
MAX_CHARS = 1500

# ---------------- Custom CSS (Cyberpunk Dark Neon Theme) ----------------
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glowing Title */
    .main-title {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    /* Tag Badges */
    .ner-token {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    .tag-PER  { background: rgba(249, 115, 22, 0.2); color: #fb923c; border-color: rgba(249, 115, 22, 0.4); }
    .tag-ORG  { background: rgba(59, 130, 246, 0.2);  color: #60a5fa; border-color: rgba(59, 130, 246, 0.4); }
    .tag-LOC  { background: rgba(34, 197, 94, 0.2);   color: #4ade80; border-color: rgba(34, 197, 94, 0.4); }
    .tag-MISC { background: rgba(168, 85, 247, 0.2);  color: #c084fc; border-color: rgba(168, 85, 247, 0.4); }

    .tag-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.4);
    }

    /* Input Styling */
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
    }

    /* Button Polish */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚡ Loading Transformer Model...")
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)

ner = load_model()

# ---------------- Session State Initializer ----------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "Elon Musk announced that Tesla will open a new facility in Berlin next year."
if "has_submitted" not in st.session_state:
    st.session_state["has_submitted"] = False

def clear_all():
    st.session_state["user_text"] = ""
    st.session_state["has_submitted"] = False

# ---------------- Sidebar Guide ----------------
with st.sidebar:
    st.header("🏷️ Entities Legend")
    st.markdown("""
    - <span style="color:#fb923c; font-weight:bold;">PER</span> : Persons / People
    - <span style="color:#60a5fa; font-weight:bold;">ORG</span> : Organizations & Companies
    - <span style="color:#4ade80; font-weight:bold;">LOC</span> : Locations, Cities, Countries
    - <span style="color:#c084fc; font-weight:bold;">MISC</span> : Miscellaneous (Nationalities, Events, etc.)
    """, unsafe_allow_html=True)
    st.divider()
    st.info("💡 Write or paste your text in the box and click **Submit** to run entity detection.")

# ---------------- UI Header ----------------
st.markdown('<div class="main-title">Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fine-tuned DistilBERT • CoNLL-2003 Standard</div>', unsafe_allow_html=True)

# Selection Option (only write your own)
st.selectbox("Input Mode", ["(write your own)"], disabled=True)

st.subheader("✍️ Write your own")
input_text = st.text_area(
    label="Enter text:",
    value=st.session_state["user_text"],
    height=140,
    placeholder="Type or paste your English text here...",
    key="text_box"
)

# Control Buttons
col1, col2, col3 = st.columns([1.5, 1.5, 7])
with col1:
    submit_btn = st.button("🚀 Submit", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", on_click=clear_all, use_container_width=True)

if submit_btn:
    st.session_state["user_text"] = input_text
    st.session_state["has_submitted"] = True

# ---------------- Output Processing & Visualization ----------------
st.divider()
st.subheader("📊 Detected Entities")

current_text = input_text.strip()[:MAX_CHARS]

if st.session_state["has_submitted"] and current_text:
    entities = sorted(ner(current_text), key=lambda e: e["start"])

    parts, rows, pos = [], [], 0
    for e in entities:
        start, end = max(e["start"], pos), e["end"]
        if start >= end:
            continue
        label = e["entity_group"]
        
        # Add preceding neutral text
        parts.append(html.escape(current_text[pos:start]))
        
        # Styled entity token badge
        parts.append(
            f'<span class="ner-token tag-{label}">'
            f'<span>{html.escape(current_text[start:end])}</span>'
            f'<span class="tag-label">{label}</span>'
            f'</span>'
        )
        
        rows.append({
            "Entity": current_text[start:end],
            "Type": label,
            "Confidence Score": f"{float(e['score']):.2%}"
        })
        pos = end

    parts.append(html.escape(current_text[pos:]))

    # Display interactive highlighted text
    st.markdown('<div style="line-height: 2.2; font-size: 1.05rem; margin-bottom: 20px;">' + "".join(parts) + '</div>', unsafe_allow_html=True)
    
    # Display table report
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No entities found in the provided text.")
else:
    st.info("Enter text above and click **Submit** to highlight entities.")
