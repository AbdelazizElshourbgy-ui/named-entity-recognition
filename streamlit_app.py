"""
Named Entity Recognition - Streamlit app

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
COLORS = {"PER": "#f97316", "ORG": "#3b82f6", "LOC": "#22c55e", "MISC": "#a855f7"}
EXAMPLES = {
    "Politics": "German chancellor Angela Merkel met French president Emmanuel Macron in Brussels to discuss the European Union budget.",
    "Football": "Real Madrid beat Manchester United 2-1 in Madrid on Saturday.",
    "Tech": "Mark Zuckerberg announced that Meta will open a new office in Cairo.",
}

st.set_page_config(page_title="Named Entity Recognition", page_icon="🏷️")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple", device=-1)


ner = load_model()

# ---------------- UI ----------------
st.title("🏷️ Named Entity Recognition")
st.write("Highlights people, organizations, locations and miscellaneous entities in English text.")

legend = " ".join(
    f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px">{k}</span>'
    for k, c in COLORS.items()
)
st.markdown(legend, unsafe_allow_html=True)

choice = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
text = st.text_area("Text", value=EXAMPLES.get(choice, ""), height=150,
                    placeholder="Type or paste an English sentence...")
text = text.strip()[:MAX_CHARS]

if text:
    entities = sorted(ner(text), key=lambda e: e["start"])

    parts, rows, pos = [], [], 0
    for e in entities:
        start, end = max(e["start"], pos), e["end"]
        if start >= end:
            continue
        label = e["entity_group"]
        parts.append(html.escape(text[pos:start]))
        parts.append(
            f'<span style="background:{COLORS.get(label, "#888")};color:white;padding:2px 6px;'
            f'border-radius:4px;margin:0 1px">{html.escape(text[start:end])} <small><b>{label}</b></small></span>'
        )
        rows.append({"Entity": text[start:end], "Type": label, "Confidence": round(float(e["score"]), 3)})
        pos = end
    parts.append(html.escape(text[pos:]))

    st.markdown(
        '<div style="line-height:2.3;font-size:1.1rem">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No entities found.")
