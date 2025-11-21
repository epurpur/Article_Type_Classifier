# Streamlit App: AI Agent to Detect Scholarly vs Popular Articles
# Uses ONLY free tools + HuggingFace models (no training required)

"""
INSTRUCTIONS
------------
1. Install dependencies:
   pip install streamlit transformers torch pdfplumber

2. Run the app:
   streamlit run app.py

3. Upload a PDF and receive classification result + confidence.

This app uses the FREE HuggingFace zero-shot model:
facebook/bart-large-mnli
"""

import streamlit as st
import pdfplumber
from transformers import pipeline

# -----------------------------
# Load AI model (cached)
# -----------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

classifier = load_classifier()

# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_text_from_pdf(uploaded_file):
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text[:4000]  # limit for model context

# -----------------------------
# Document Classification
# -----------------------------

def classify_text(text):
    labels = [
        "scholarly peer-reviewed academic article",
        "popular or non-academic article"
    ]

    result = classifier(text, labels)
    return result

# -----------------------------
# Heuristic Evidence (Explainability)
# -----------------------------

def scholarly_indicators(text):
    indicators = {
        "Contains Abstract": "Abstract" in text,
        "Contains References": "References" in text or "Bibliography" in text,
        "Contains DOI": "doi" in text.lower(),
        "Mentions Journal/Volume": "Journal" in text or "Vol." in text
    }
    return indicators

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📄 AI Scholarly Article Detector")
st.markdown("Upload a PDF and this AI will determine if it is peer-reviewed or a popular source.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Analyzing document..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        result = classify_text(extracted_text)
        heuristics = scholarly_indicators(extracted_text)

    predicted_label = result['labels'][0]
    confidence = result['scores'][0]

    st.subheader("📊 Classification Result")

    if "scholarly" in predicted_label:
        st.success(f"✅ Scholarly / Peer-reviewed (Confidence: {confidence:.2%})")
    else:
        st.warning(f"⚠️ Popular / Non-academic (Confidence: {confidence:.2%})")

    st.subheader("🔍 Evidence Detected")
    for key, value in heuristics.items():
        icon = "✅" if value else "❌"
        st.write(f"{icon} {key}")

    with st.expander("🔎 Show Extracted Text"):
        st.text(extracted_text[:2000])

