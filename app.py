"""
Named Entity Recognition - Gradio app

Loads the fine-tuned DistilBERT model saved by the notebook (ner_model/ folder)
and highlights people (PER), organizations (ORG), locations (LOC) and
miscellaneous entities (MISC) with colors.

Run:
    pip install gradio transformers torch
    python app.py
"""
import os

# On Hugging Face ZeroGPU Spaces the `spaces` package must be imported BEFORE torch.
# Locally (or on a normal CPU Space) it is optional, so we fall back to a no-op decorator.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

import gradio as gr
import torch
from transformers import pipeline

# ---------------- Config ----------------
# Folder saved by the notebook (trainer.save_model + tokenizer.save_pretrained).
# Override with: MODEL_DIR=/path/to/ner_model python app.py
CANDIDATE_DIRS = [
    os.environ.get("MODEL_DIR", ""),
    "ner_model",
    "/kaggle/working/ner_model",
]
MODEL_DIR = next((d for d in CANDIDATE_DIRS if d and os.path.isdir(d)), None)
if MODEL_DIR is None:
    raise FileNotFoundError(
        "Model folder not found. Put the 'ner_model' folder next to app.py "
        "or set the MODEL_DIR environment variable."
    )

MAX_CHARS = 1500
COLORS = {"PER": "#f97316", "ORG": "#3b82f6", "LOC": "#22c55e", "MISC": "#a855f7"}

# ---------------- Load model ----------------
ner = pipeline(
    "token-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple",          # merges word pieces and B-/I- tags into whole entities
    device=0 if torch.cuda.is_available() else -1,
)


# ---------------- Inference ----------------
@gpu
def predict(text: str):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return [], []

    entities = sorted(ner(text), key=lambda e: e["start"])

    segments, table, pos = [], [], 0
    for e in entities:
        start, end = max(e["start"], pos), e["end"]
        if start >= end:
            continue
        if start > pos:
            segments.append((text[pos:start], None))
        segments.append((text[start:end], e["entity_group"]))
        table.append([text[start:end], e["entity_group"], round(float(e["score"]), 3)])
        pos = end
    if pos < len(text):
        segments.append((text[pos:], None))
    return segments, table


# ---------------- UI ----------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=5,
        label="Text",
        placeholder="Type or paste an English sentence...",
    ),
    outputs=[
        gr.HighlightedText(label="Detected entities", color_map=COLORS, show_legend=True, combine_adjacent=False),
        gr.Dataframe(headers=["Entity", "Type", "Confidence"], label="Entities table"),
    ],
    title="Named Entity Recognition",
    description="Highlights people (PER), organizations (ORG), locations (LOC) and miscellaneous entities (MISC) in English text.",
    examples=[
        ["Barack Obama visited Google headquarters in California last Tuesday."],
        ["Real Madrid beat Manchester United 2-1 in Madrid on Saturday."],
        ["Mark Zuckerberg announced that Meta will open a new office in Cairo."],
    ],
)

if __name__ == "__main__":
    demo.launch()
