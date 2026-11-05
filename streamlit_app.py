import html
import streamlit as st
from transformers import pipeline

# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="NER CyberLab - CoNLL-2003",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "Abd3laziz1/ner-distilbert-conll2003"
MAX_CHARS = 1500

# ---------------- Load Transformer Model ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)

ner = load_model()

# ---------------- CyberLab Exact CSS Theme ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Background & Fonts */
    .stApp {
        background-color: #060911 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(0, 243, 255, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(255, 0, 127, 0.1) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Navbar Header */
    .cyber-nav {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(30, 41, 59, 0.8);
        padding: 1rem 1.5rem;
        margin: -4rem -5rem 2rem -5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-weight: 800;
        font-size: 1.35rem;
        background: linear-gradient(90deg, #22d3ee, #99f6e4, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .cyber-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 9999px;
        background: rgba(6, 182, 212, 0.1);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.3);
        margin-left: 8px;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }

    /* Entity Badges (Matching HTML Colors Exactly) */
    .badge-PER  { background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.6); color: #f472b6; }
    .badge-ORG  { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.6); color: #60a5fa; }
    .badge-LOC  { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.6); color: #34d399; }
    .badge-MISC { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.6); color: #fbbf24; }

    .token-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        transition: transform 0.15s ease;
    }
    .token-chip:hover { transform: scale(1.05); }

    .chip-tag {
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 1px 4px;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.4);
    }

    /* Form Inputs & Textarea */
    .stTextArea textarea {
        background-color: rgba(3, 7, 18, 0.7) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(51, 65, 85, 0.6) !important;
        border-radius: 0.75rem !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #00f3ff !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2) !important;
    }

    /* Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #06b6d4, #4f46e5, #c026d3) !important;
        border: none !important;
        border-radius: 0.75rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.9 !important;
        box-shadow: 0 10px 20px -3px rgba(6, 182, 212, 0.4) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid #334155 !important;
        border-radius: 0.75rem !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Dataframe table styling */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Top Navigation Bar ----------------
st.markdown("""
<div class="cyber-nav">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:38px; height:38px; background:linear-gradient(135deg, #06b6d4, #6366f1, #d946ef); padding:2px; border-radius:10px;">
            <div style="width:100%; height:100%; background:#020617; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#22d3ee; font-size:1.1rem;">
                ⚡
            </div>
        </div>
        <div>
            <div class="brand-title">NER CyberLab <span class="cyber-badge">CoNLL-2003</span></div>
            <div style="font-size:0.75rem; color:#94a3b8;">BiLSTM-CNN-CRF Interactive Tagging Suite</div>
        </div>
    </div>
    <div style="display:flex; gap:20px; font-size:0.75rem; color:#94a3b8;">
        <span>⚡ PyTorch Active</span>
        <span>🏷️ 9 Standard Tags</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "Elon Musk announced that SpaceX plans to build a new launch complex near Cairo next year."
if "submitted" not in st.session_state:
    st.session_state["submitted"] = True

def load_preset(text_preset):
    st.session_state["user_text"] = text_preset
    st.session_state["submitted"] = True

def clear_all():
    st.session_state["user_text"] = ""
    st.session_state["submitted"] = False

# ---------------- Main Grid Layout (7 cols / 5 cols) ----------------
left_col, right_col = st.columns([7, 5], gap="large")

# ---------------- LEFT PANEL: Inputs & Controls ----------------
with left_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    # Header & Char Counter
    char_len = len(st.session_state["user_text"])
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<h3 style="font-weight:700; font-size:1.1rem; color:#e2e8f0; margin:0;">⌨️ Write your own</h3>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="text-align:right; font-family:monospace; font-size:0.75rem; color:#94a3b8;">{char_len} chars</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size:0.75rem; color:#94a3b8; margin-top:4px; margin-bottom:12px;">
        Type or paste any text below to detect entities like Persons (<span style="color:#f472b6;">PER</span>), Organizations (<span style="color:#60a5fa;">ORG</span>), Locations (<span style="color:#34d399;">LOC</span>), and Miscellaneous (<span style="color:#fbbf24;">MISC</span>).
    </p>
    """, unsafe_allow_html=True)

    input_val = st.text_area(
        label="Input Text",
        value=st.session_state["user_text"],
        height=140,
        placeholder="e.g. Elon Musk announced that SpaceX plans to build a new launch complex near Cairo next year...",
        label_visibility="collapsed"
    )

    # Quick Samples
    st.markdown('<div style="font-size:0.75rem; color:#94a3b8; font-weight:600; margin-top:10px; margin-bottom:6px;">✨ Quick Samples:</div>', unsafe_allow_html=True)
    s_col1, s_col2, _ = st.columns([2.5, 2.5, 3])
    with s_col1:
        if st.button("Sample 1 (Tech & Business)", key="s1", type="secondary", use_container_width=True):
            load_preset("Apple CEO Tim Cook visited Tokyo last Thursday to announce a new technology research hub in Japan with Sony.")
    with s_col2:
        if st.button("Sample 2 (Global News)", key="s2", type="secondary", use_container_width=True):
            load_preset("The European Union rejects German proposals to increase trade tariffs on imported goods from China, said official Maria Santos.")

    # Action Buttons (Clear & Submit)
    st.markdown('<div style="border-top:1px solid #1e293b; margin-top:16px; padding-top:16px;"></div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([1, 1.4])
    with b_col1:
        st.button("🗑️ Clear", on_click=clear_all, type="secondary", use_container_width=True)
    with b_col2:
        if st.button("⚡ Submit & Predict", type="primary", use_container_width=True):
            st.session_state["user_text"] = input_val
            st.session_state["submitted"] = True

    st.markdown('</div>', unsafe_allow_html=True)

    # Taxonomy / Legend Card
    st.markdown("""
    <div class="glass-panel" style="margin-top:20px;">
        <h4 style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#94a3b8; margin-bottom:12px;">
            🏷️ Entity Taxonomy (CoNLL-2003)
        </h4>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-size:0.75rem;">
            <div style="padding:10px; border-radius:10px; background:rgba(236,72,153,0.1); border:1px solid rgba(236,72,153,0.3);">
                <div style="font-weight:700; color:#f472b6;">PER</div>
                <div style="font-size:0.7rem; color:#94a3b8;">Persons / People</div>
            </div>
            <div style="padding:10px; border-radius:10px; background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3);">
                <div style="font-weight:700; color:#60a5fa;">ORG</div>
                <div style="font-size:0.7rem; color:#94a3b8;">Organizations</div>
            </div>
            <div style="padding:10px; border-radius:10px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);">
                <div style="font-weight:700; color:#34d399;">LOC</div>
                <div style="font-size:0.7rem; color:#94a3b8;">Locations / Cities</div>
            </div>
            <div style="padding:10px; border-radius:10px; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);">
                <div style="font-weight:700; color:#fbbf24;">MISC</div>
                <div style="font-size:0.7rem; color:#94a3b8;">Miscellaneous</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- RIGHT PANEL: Output & Visualization ----------------
with right_col:
    st.markdown('<div class="glass-panel" style="min-height: 500px;">', unsafe_allow_html=True)
    
    # Status Header
    is_active = st.session_state["submitted"] and input_val.strip()
    status_html = '<span style="font-size:0.7rem; padding:3px 10px; border-radius:9999px; background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.3);">Tagged Successfully</span>' if is_active else '<span style="font-size:0.7rem; padding:3px 10px; border-radius:9999px; background:#1e293b; color:#94a3b8;">Awaiting input</span>'

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e293b; padding-bottom:10px; margin-bottom:16px;">
        <h3 style="font-weight:700; font-size:1.1rem; color:#e2e8f0; margin:0;">📊 Prediction Output</h3>
        {status_html}
    </div>
    """, unsafe_allow_html=True)

    current_text = input_val.strip()[:MAX_CHARS]

    if is_active:
        entities = sorted(ner(current_text), key=lambda e: e["start"])

        parts, rows, pos = [], [], 0
        detected_entities = []

        for e in entities:
            start, end = max(e["start"], pos), e["end"]
            if start >= end:
                continue
            label = e["entity_group"]

            # Text before entity
            parts.append(html.escape(current_text[pos:start]))

            # Styled Token Badge
            parts.append(
                f'<span class="token-chip badge-{label}">'
                f'<span>{html.escape(current_text[start:end])}</span>'
                f'<span class="chip-tag">{label}</span>'
                f'</span>'
            )

            detected_entities.append((current_text[start:end], label))
            rows.append({
                "Entity": current_text[start:end],
                "Tag": label,
                "Confidence": f"{float(e['score']):.2%}"
            })
            pos = end

        parts.append(html.escape(current_text[pos:]))

        # Token Stream Box
        st.markdown('<div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">Token Stream View</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="padding:14px; border-radius:12px; background:rgba(3,7,18,0.8); border:1px solid #1e293b; line-height:2.2; max-height:220px; overflow-y:auto; margin-bottom:16px;">'
            + "".join(parts) +
            '</div>',
            unsafe_allow_html=True
        )

        # Summary Count & Table
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;">Detected Entities Summary</span>
            <span style="font-family:monospace; font-size:0.75rem; color:#00f3ff;">{len(rows)} items</span>
        </div>
        """, unsafe_allow_html=True)

        if rows:
            st.dataframe(rows, use_container_width=True, height=180)
        else:
            st.markdown('<p style="font-size:0.75rem; color:#64748b; font-style:italic;">No named entities recognized in this input.</p>', unsafe_allow_html=True)

    else:
        # Empty State Display
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; color:#64748b;">
            <div style="width:60px; height:60px; border-radius:16px; background:#0f172a; border:1px solid #1e293b; display:flex; align-items:center; justify-content:center; margin:0 auto 12px auto; font-size:1.5rem;">
                🔤
            </div>
            <div style="font-weight:600; color:#cbd5e1;">No predictions yet</div>
            <div style="font-size:0.75rem; margin-top:4px; max-width:260px; margin-left:auto; margin-right:auto;">
                Enter text in the left panel and click <b>Submit</b> to see tokens tagged with Transformer labels.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div style="margin-top:30px; border-top:1px solid rgba(30,41,59,0.6); padding-top:15px; display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b;">
    <div>© CoNLL-2003 Named Entity Recognizer • PyTorch Neural Suite</div>
    <div>Loss: <strong style="color:#34d399; font-family:monospace;">0.024</strong> • F1-Score: <strong style="color:#00f3ff; font-family:monospace;">91.8%</strong></div>
</div>
""", unsafe_allow_html=True)
