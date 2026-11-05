import streamlit as st
import html
from transformers import pipeline

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="NER CyberLab - CoNLL-2003",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit Default Elements & Margins
st.markdown("""
<style>
    .block-container { padding: 0rem !important; max-width: 100% !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

MODEL_ID = "Abd3laziz1/ner-distilbert-conll2003"

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)

ner = load_model()

# ---------------- Session State Initializer ----------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "Elon Musk announced that SpaceX plans to build a new launch complex near Cairo next year."
if "submitted" not in st.session_state:
    st.session_state["submitted"] = True

# Callback functions for buttons
def handle_submit():
    st.session_state["submitted"] = True

def handle_clear():
    st.session_state["user_text"] = ""
    st.session_state["submitted"] = False

def load_sample(sample_text):
    st.session_state["user_text"] = sample_text
    st.session_state["submitted"] = True

# ---------------- Exact HTML/Tailwind CyberLab Design (No Top Nav) ----------------
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    .cyber-body {
        background-color: #060911;
        background-image: 
            radial-gradient(at 0% 0%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(0, 243, 255, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(255, 0, 127, 0.1) 0px, transparent 50%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f1f5f9;
        min-height: 100vh;
        padding: 2rem;
    }

    .glass-panel {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
    }

    .badge-PER { background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.6); color: #f472b6; }
    .badge-ORG { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.6); color: #60a5fa; }
    .badge-LOC { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.6); color: #34d399; }
    .badge-MISC { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.6); color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

# Main Container Layout (Grid 7/5 Cols matching HTML)
st.markdown('<div class="cyber-body">', unsafe_allow_html=True)
col_left, col_right = st.columns([7, 5], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-panel p-6 shadow-2xl relative overflow-hidden mb-6">
        <div class="flex items-center justify-between mb-3">
            <label class="flex items-center gap-2 font-bold text-lg text-slate-200">
                <i class="fa-solid fa-keyboard text-cyan-400"></i> Write your own
            </label>
        </div>
        <p class="text-xs text-slate-400 mb-3">
            Type or paste any text below to detect entities like Persons (<span class="text-pink-400">PER</span>), Organizations (<span class="text-blue-400">ORG</span>), Locations (<span class="text-emerald-400">LOC</span>), and Miscellaneous (<span class="text-amber-400">MISC</span>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit Input bound to session state
    user_input = st.text_area(
        label="Input Text",
        value=st.session_state["user_text"],
        height=140,
        placeholder="e.g. Elon Musk announced that SpaceX plans to build a new launch complex near Cairo next year...",
        label_visibility="collapsed"
    )
    st.session_state["user_text"] = user_input

    # Quick Samples & Buttons inside Streamlit columns
    st.markdown('<div class="flex items-center gap-2 mt-3 mb-4 text-xs text-slate-400"><i class="fa-solid fa-wand-magic-sparkles text-fuchsia-400"></i> <b>Quick Samples:</b></div>', unsafe_allow_html=True)
    sc1, sc2, _ = st.columns([2, 2, 3])
    with sc1:
        if st.button("Sample 1 (Tech & Business)", use_container_width=True):
            load_sample("Apple CEO Tim Cook visited Tokyo last Thursday to announce a new technology research hub in Japan with Sony.")
            st.rerun()
    with sc2:
        if st.button("Sample 2 (Global News)", use_container_width=True):
            load_sample("The European Union rejects German proposals to increase trade tariffs on imported goods from China, said official Maria Santos.")
            st.rerun()

    # Action Buttons Row
    bc1, bc2 = st.columns([1, 1])
    with bc1:
        st.button("🗑️ Clear", on_click=handle_clear, use_container_width=True)
    with bc2:
        st.button("⚡ Submit & Predict", on_click=handle_submit, type="primary", use_container_width=True)

    # Taxonomy / Legend Card
    st.markdown("""
    <div class="glass-panel p-5 shadow-xl mt-6">
        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
            <i class="fa-solid fa-tags text-cyan-400"></i> Entity Taxonomy (CoNLL-2003)
        </h3>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div class="p-2.5 rounded-xl bg-pink-950/20 border border-pink-500/30 flex flex-col gap-1">
                <span class="font-bold text-pink-400 flex items-center justify-between">PER <i class="fa-solid fa-user text-[10px]"></i></span>
                <span class="text-slate-400 text-[11px]">Persons / People</span>
            </div>
            <div class="p-2.5 rounded-xl bg-blue-950/20 border border-blue-500/30 flex flex-col gap-1">
                <span class="font-bold text-blue-400 flex items-center justify-between">ORG <i class="fa-solid fa-building text-[10px]"></i></span>
                <span class="text-slate-400 text-[11px]">Organizations</span>
            </div>
            <div class="p-2.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 flex flex-col gap-1">
                <span class="font-bold text-emerald-400 flex items-center justify-between">LOC <i class="fa-solid fa-location-dot text-[10px]"></i></span>
                <span class="text-slate-400 text-[11px]">Locations / Cities</span>
            </div>
            <div class="p-2.5 rounded-xl bg-amber-950/20 border border-amber-500/30 flex flex-col gap-1">
                <span class="font-bold text-amber-400 flex items-center justify-between">MISC <i class="fa-solid fa-globe text-[10px]"></i></span>
                <span class="text-slate-400 text-[11px]">Miscellaneous</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-panel p-6 shadow-2xl flex flex-col min-h-[480px]">', unsafe_allow_html=True)
    
    current_text = user_input.strip()[:1500]
    is_active = st.session_state["submitted"] and current_text

    status_badge = '<span class="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold">Tagged Successfully</span>' if is_active else '<span class="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700">Awaiting input</span>'

    st.markdown(f"""
    <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <h2 class="font-bold text-lg text-slate-200 flex items-center gap-2">
            <i class="fa-solid fa-square-poll-vertical text-fuchsia-400"></i> Prediction Output
        </h2>
        {status_badge}
    </div>
    """, unsafe_allow_html=True)

    if is_active:
        entities = sorted(ner(current_text), key=lambda e: e["start"])
        parts, rows, pos = [], [], 0

        for e in entities:
            start, end = max(e["start"], pos), e["end"]
            if start >= end:
                continue
            label = e["entity_group"]
            parts.append(html.escape(current_text[pos:start]))
            parts.append(
                f'<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono font-medium badge-{label} m-1">'
                f'<span>{html.escape(current_text[start:end])}</span>'
                f'<span class="text-[9px] opacity-70 font-bold uppercase px-1 rounded bg-black/20">{label}</span>'
                f'</span>'
            )
            rows.append({"Entity": current_text[start:end], "Type": label, "Confidence": f"{float(e['score']):.2%}"})
            pos = end
        parts.append(html.escape(current_text[pos:]))

        st.markdown('<h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Token Stream View</h4>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-wrap gap-1 leading-loose min-h-[140px] max-h-[220px] overflow-y-auto mb-4">'
            + "".join(parts) +
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex justify-between items-center">
            <span>Detected Entities Summary</span>
            <span class="text-cyan-400 font-mono text-xs">{len(rows)} items</span>
        </h4>
        """, unsafe_allow_html=True)

        if rows:
            st.dataframe(rows, use_container_width=True, height=180)
        else:
            st.markdown('<p class="text-xs text-slate-500 italic p-2">No named entities recognized in this input.</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="my-auto flex flex-col items-center justify-center text-center p-12 text-slate-500">
            <div class="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-4 text-slate-600 text-2xl">
                <i class="fa-solid fa-font"></i>
            </div>
            <p class="font-semibold text-slate-300">No predictions yet</p>
            <p class="text-xs mt-1 max-w-xs text-slate-400">Enter text in the left panel and click <b>Submit & Predict</b> to see tokens tagged.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
