"""
Named Entity Recognition - Streamlit app (CyberLab design)

The fine-tuned DistilBERT model is loaded from the Hugging Face Hub (a MODEL repo),
so the model file does not need to be stored in GitHub.

Run locally:
    pip install streamlit transformers torch
    streamlit run streamlit_app.py
"""
import html

import streamlit as st
from transformers import pipeline

# ---------------- Config ----------------
MODEL_ID = "Abd3laziz1/ner-distilbert-conll2003"   # <- your model repo on the Hugging Face Hub
MAX_CHARS = 1500

TAGS = {
    "PER":  {"desc": "Persons / People",  "icon": "fa-user",         "fg": "#f472b6", "rgb": "236,72,153"},
    "ORG":  {"desc": "Organizations",     "icon": "fa-building",     "fg": "#60a5fa", "rgb": "59,130,246"},
    "LOC":  {"desc": "Locations / Cities", "icon": "fa-location-dot", "fg": "#34d399", "rgb": "16,185,129"},
    "MISC": {"desc": "Miscellaneous",     "icon": "fa-globe",        "fg": "#fbbf24", "rgb": "245,158,11"},
}
SAMPLES = {
    "Sample 1 (Tech & Business)": "Apple CEO Tim Cook visited Tokyo last Thursday to announce a new technology research hub in Japan with Sony.",
    "Sample 2 (Global News)": "The European Union rejects German proposals to increase trade tariffs on imported goods from China, said official Maria Santos.",
    "Sample 3 (Football)": "Real Madrid beat Manchester United 2-1 in Madrid on Saturday.",
}

st.set_page_config(page_title="NER CyberLab", page_icon="🏷️", layout="wide", initial_sidebar_state="collapsed")

# ---------------- Styling ----------------
tag_css = "".join(
    f".b-{k}{{background:rgba({v['rgb']},.15);border:1px solid rgba({v['rgb']},.6);color:{v['fg']};}}"
    f".c-{k}{{background:rgba({v['rgb']},.08);border:1px solid rgba({v['rgb']},.3);color:{v['fg']};}}"
    for k, v in TAGS.items()
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
/* ---- Hide Streamlit chrome (Share / star / edit / GitHub / menu / footer) ---- */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stAppToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
.stAppDeployButton, #MainMenu, footer, [class*="viewerBadge"], [data-testid="stSidebarCollapsedControl"]
{display:none !important;visibility:hidden !important;height:0 !important;}
/* ---- Page ---- */
html, body, .stApp, [class*="st-"], button, textarea, input {font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{
background-color:#060911;
background-image:radial-gradient(at 0% 0%, rgba(168,85,247,.15) 0px, transparent 50%),
radial-gradient(at 100% 0%, rgba(0,243,255,.15) 0px, transparent 50%),
radial-gradient(at 50% 100%, rgba(255,0,127,.10) 0px, transparent 50%);
background-attachment:fixed;color:#f1f5f9;}
.block-container{max-width:80rem;padding:0 1.5rem 2rem 1.5rem !important;}
::selection{background:#06b6d4;color:#000;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:rgba(15,23,42,.6);}
::-webkit-scrollbar-thumb{background:rgba(0,243,255,.3);border-radius:4px;}
/* ---- Glass panels ---- */
.glass{background:rgba(15,23,42,.75);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.08);}
.glass:hover{border-color:rgba(0,243,255,.3);}
div[data-testid="stVerticalBlockBorderWrapper"]{
background:rgba(15,23,42,.75);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
border:1px solid rgba(255,255,255,.08) !important;border-radius:16px !important;
padding:.9rem .6rem;box-shadow:0 25px 50px -12px rgba(0,0,0,.5);transition:border-color .3s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{border-color:rgba(0,243,255,.3) !important;}
/* ---- Navbar ---- */
.nav{border-bottom:1px solid rgba(30,41,59,.8);border-radius:0 0 16px 16px;padding:1rem 1.5rem;margin:0 0 1.5rem 0;
display:flex;align-items:center;justify-content:space-between;}
.logo{width:40px;height:40px;border-radius:12px;padding:2px;background:linear-gradient(45deg,#06b6d4,#6366f1,#d946ef);
box-shadow:0 10px 15px -3px rgba(6,182,212,.2);}
.logo>div{width:100%;height:100%;background:#020617;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#22d3ee;font-size:1.1rem;}
.nav h1{margin:0;padding:0;font-weight:800;font-size:1.25rem;letter-spacing:-.025em;line-height:1.3;
background:linear-gradient(90deg,#22d3ee,#99f6e4,#e879f9);-webkit-background-clip:text;background-clip:text;color:transparent;}
.pill-cyan{font-size:.72rem;padding:2px 8px;border-radius:999px;background:rgba(6,182,212,.1);color:#22d3ee;
border:1px solid rgba(6,182,212,.3);-webkit-text-fill-color:#22d3ee;font-weight:500;margin-left:6px;}
.nav p{margin:0;font-size:.75rem;color:#94a3b8;}
.nav-right{display:flex;gap:1.5rem;font-size:.75rem;color:#94a3b8;}
.nav-right i{margin-right:.4rem;}
@media (max-width:768px){.nav-right{display:none;}}
/* ---- Card headings ---- */
.card-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem;}
.card-title .t{font-weight:700;font-size:1.1rem;color:#e2e8f0;display:flex;align-items:center;gap:.5rem;}
.mono{font-family:'JetBrains Mono',monospace;}
.muted{color:#94a3b8;font-size:.75rem;}
.sec-label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;margin:.2rem 0 .5rem 0;
display:flex;justify-content:space-between;align-items:center;}
/* ---- Textarea ---- */
.stTextArea textarea{background:rgba(2,6,23,.7) !important;color:#f1f5f9 !important;border-radius:12px !important;
border:1px solid rgba(51,65,85,.6) !important;padding:1rem !important;line-height:1.6;font-size:1rem;}
.stTextArea textarea::placeholder{color:#475569;}
.stTextArea textarea:focus{border-color:#22d3ee !important;box-shadow:0 0 0 2px rgba(34,211,238,.2) !important;}
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"]{background:transparent !important;border:none !important;}
.stTextArea label{display:none;}
/* ---- Buttons ---- */
.stButton>button, [data-testid^="stBaseButton"]{border-radius:12px;font-weight:600;font-size:.85rem;transition:all .2s;}
button[data-testid="stBaseButton-secondary"], .stButton>button[kind="secondary"]{
background:rgba(30,41,59,.8);color:#cbd5e1;border:1px solid #334155;}
button[data-testid="stBaseButton-secondary"]:hover, .stButton>button[kind="secondary"]:hover{
background:#334155;border-color:rgba(34,211,238,.4);color:#fff;}
button[data-testid="stBaseButton-primary"], .stButton>button[kind="primary"]{
background:linear-gradient(90deg,#06b6d4,#4f46e5,#c026d3);color:#fff;border:none;font-weight:700;
box-shadow:0 10px 15px -3px rgba(6,182,212,.25);}
button[data-testid="stBaseButton-primary"]:hover, .stButton>button[kind="primary"]:hover{
background:linear-gradient(90deg,#22d3ee,#6366f1,#d946ef);box-shadow:0 10px 15px -3px rgba(6,182,212,.4);color:#fff;}
.stButton>button:active{transform:scale(.97);}
/* ---- Legend ---- */
.legend{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;}
@media (max-width:768px){.legend{grid-template-columns:repeat(2,1fr);}}
.lg{padding:.65rem;border-radius:12px;display:flex;flex-direction:column;gap:.25rem;font-size:.75rem;}
.lg b{display:flex;justify-content:space-between;align-items:center;}
.lg span{color:#94a3b8;font-size:.7rem;}
/* ---- Output ---- */
.status{font-size:.75rem;padding:4px 10px;border-radius:999px;background:#1e293b;color:#94a3b8;border:1px solid #334155;}
.status.ok{background:rgba(16,185,129,.1);color:#34d399;border-color:rgba(16,185,129,.3);font-weight:600;}
.divider{border-bottom:1px solid #1e293b;margin:.2rem 0 1rem 0;}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:3rem 1rem;color:#64748b;}
.empty .ic{width:64px;height:64px;border-radius:16px;background:#0f172a;border:1px solid #1e293b;
display:flex;align-items:center;justify-content:center;margin-bottom:1rem;color:#475569;font-size:1.5rem;}
.empty p{margin:0;}
.stream{padding:1rem;border-radius:12px;background:rgba(2,6,23,.8);border:1px solid #1e293b;line-height:2.4;
min-height:120px;max-height:240px;overflow-y:auto;color:#cbd5e1;font-size:.95rem;margin-bottom:1.2rem;}
.badge{display:inline-flex;align-items:center;gap:.4rem;padding:2px 10px;border-radius:8px;font-size:.8rem;
font-family:'JetBrains Mono',monospace;font-weight:500;margin:0 2px;transition:transform .15s;}
.badge:hover{transform:scale(1.05);}
.badge small{font-size:.6rem;opacity:.75;font-weight:700;text-transform:uppercase;padding:0 4px;border-radius:4px;background:rgba(0,0,0,.25);}
.ents{display:flex;flex-direction:column;gap:.5rem;max-height:260px;overflow-y:auto;padding-right:4px;}
.ent{display:flex;align-items:center;justify-content:space-between;padding:.6rem;border-radius:12px;font-size:.78rem;}
.ent .n{font-weight:700;display:flex;align-items:center;gap:.5rem;}
.ent .r{display:flex;align-items:center;gap:.5rem;}
.conf{font-family:'JetBrains Mono',monospace;color:#94a3b8;font-size:.68rem;}
.tg{font-family:'JetBrains Mono',monospace;font-weight:800;font-size:.62rem;padding:2px 8px;border-radius:4px;background:rgba(0,0,0,.4);}
.noent{font-size:.75rem;color:#64748b;font-style:italic;padding:.5rem;}
/* ---- Footer ---- */
.foot{border-top:1px solid rgba(30,41,59,.6);border-radius:16px 16px 0 0;padding:1rem 1.5rem;margin-top:1.5rem;
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;font-size:.75rem;color:#64748b;}
/* ---- Misc ---- */
.stAlert{border-radius:12px;}
hr{border-color:#1e293b;}
""" + tag_css + """
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- Model ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)


ner = load_model()

# ---------------- State ----------------
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("result", None)


def load_sample(name):
    st.session_state.text_input = SAMPLES[name]
    st.session_state.result = None


def clear_input():
    st.session_state.text_input = ""
    st.session_state.result = None


# ---------------- Navbar ----------------
st.markdown(
    '<div class="glass nav">'
    '<div style="display:flex;align-items:center;gap:.75rem">'
    '<div class="logo"><div><i class="fa-solid fa-microchip"></i></div></div>'
    '<div><h1>NER CyberLab <span class="pill-cyan">CoNLL-2003</span></h1>'
    '<p>DistilBERT Interactive Tagging Suite</p></div></div>'
    '<div class="nav-right">'
    '<span><i class="fa-solid fa-bolt" style="color:#fbbf24"></i>Hugging Face Transformers</span>'
    '<span><i class="fa-solid fa-layer-group" style="color:#22d3ee"></i>4 Entity Types</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([7, 5], gap="large")

# ---------------- Left panel ----------------
with left:
    with st.container(border=True):
        n_chars = len(st.session_state.text_input)
        st.markdown(
            '<div class="card-title"><span class="t"><i class="fa-solid fa-keyboard" style="color:#22d3ee"></i>Write your own</span>'
            f'<span class="mono muted">{n_chars} chars</span></div>'
            '<p class="muted" style="margin-bottom:.7rem">Type or paste any English text below to detect entities like '
            'Persons (<span style="color:#f472b6">PER</span>), Organizations (<span style="color:#60a5fa">ORG</span>), '
            'Locations (<span style="color:#34d399">LOC</span>), and Miscellaneous (<span style="color:#fbbf24">MISC</span>).</p>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Text", key="text_input", height=160, max_chars=MAX_CHARS, label_visibility="collapsed",
            placeholder="e.g. Elon Musk announced that SpaceX plans to build a new launch complex near Cairo next year...",
        )

        st.markdown(
            '<div class="muted" style="font-weight:600;margin:.2rem 0 .3rem 0">'
            '<i class="fa-solid fa-wand-magic-sparkles" style="color:#e879f9"></i> Quick Samples:</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SAMPLES))
        for col, name in zip(cols, SAMPLES):
            col.button(name, key=f"s_{name}", on_click=load_sample, args=(name,), use_container_width=True)

        st.markdown('<div class="divider" style="margin-top:.8rem"></div>', unsafe_allow_html=True)
        _, c_clear, c_submit = st.columns([4, 1.4, 2.2])
        c_clear.button("🗑️ Clear", on_click=clear_input, use_container_width=True)
        submit = c_submit.button("⚡ Submit & Predict", type="primary", use_container_width=True)

    if submit:
        txt = st.session_state.text_input.strip()[:MAX_CHARS]
        if not txt:
            st.warning("Please enter some text in the text area first!")
        else:
            with st.spinner("Running DistilBERT model..."):
                ents = sorted(ner(txt), key=lambda e: e["start"])
            st.session_state.result = {"text": txt, "entities": ents}

    legend = "".join(
        f'<div class="lg c-{k}"><b>{k} <i class="fa-solid {v["icon"]}" style="font-size:.65rem"></i></b><span>{v["desc"]}</span></div>'
        for k, v in TAGS.items()
    )
    st.markdown(
        '<div class="glass" style="border-radius:16px;padding:1.2rem;margin-top:1rem">'
        '<div class="sec-label" style="justify-content:flex-start;gap:.5rem">'
        '<i class="fa-solid fa-tags" style="color:#22d3ee"></i>Entity Taxonomy (CoNLL-2003)</div>'
        f'<div class="legend">{legend}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------- Right panel ----------------
with right:
    with st.container(border=True):
        res = st.session_state.result
        status = (
            '<span class="status ok">Tagged Successfully</span>' if res
            else '<span class="status">Awaiting input</span>'
        )
        st.markdown(
            '<div class="card-title"><span class="t"><i class="fa-solid fa-square-poll-vertical" style="color:#e879f9"></i>'
            f'Prediction Output</span>{status}</div><div class="divider"></div>',
            unsafe_allow_html=True,
        )

        if not res:
            st.markdown(
                '<div class="empty"><div class="ic"><i class="fa-solid fa-font"></i></div>'
                '<p style="font-weight:600;color:#cbd5e1">No predictions yet</p>'
                '<p class="muted" style="margin-top:.3rem;max-width:20rem">Enter text in the left panel and click '
                '<b>Submit &amp; Predict</b> to see the entities tagged by the model.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            text, entities = res["text"], res["entities"]
            parts, items, pos = [], [], 0
            for e in entities:
                start, end = max(e["start"], pos), e["end"]
                if start >= end:
                    continue
                label = e["entity_group"]
                name = text[start:end]
                score = float(e["score"])
                parts.append(html.escape(text[pos:start]))
                parts.append(
                    f'<span class="badge b-{label}" title="Confidence: {score:.3f}">'
                    f'<span>{html.escape(name)}</span><small>{label}</small></span>'
                )
                items.append((name, label, score))
                pos = end
            parts.append(html.escape(text[pos:]))

            st.markdown(
                '<div class="sec-label">Token Stream View</div>'
                f'<div class="stream">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )

            if items:
                rows = ""
                for name, label, score in items:
                    t = TAGS.get(label, TAGS["MISC"])
                    rows += (
                        f'<div class="ent c-{label if label in TAGS else "MISC"}">'
                        f'<span class="n"><i class="fa-solid {t["icon"]}"></i>{html.escape(name)}</span>'
                        f'<span class="r"><span class="conf">{score * 100:.1f}%</span><span class="tg">{html.escape(label)}</span></span></div>'
                    )
                body = f'<div class="ents">{rows}</div>'
            else:
                body = '<div class="noent">No named entities (PER, ORG, LOC, MISC) recognized in this input.</div>'

            st.markdown(
                '<div class="sec-label"><span>Detected Entities Summary</span>'
                f'<span class="mono" style="color:#22d3ee">{len(items)} detected</span></div>{body}',
                unsafe_allow_html=True,
            )

# ---------------- Footer ----------------
st.markdown(
    '<div class="glass foot"><span>© CoNLL-2003 Named Entity Recognizer &bull; DistilBERT Fine-tuned Model</span>'
    f'<span class="mono">{html.escape(MODEL_ID)}</span></div>',
    unsafe_allow_html=True,
)
