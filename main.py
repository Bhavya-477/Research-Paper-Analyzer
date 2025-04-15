import os
import tempfile
import streamlit as st
import torch
import pdfplumber
import pyttsx3
from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- UTILITIES ----------
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_via_pdfplumber(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def summarize_via_model(text: str, tokenizer, model) -> str:
    if not text.strip():
        return "No text available for summarization."
    
    short_text = text[:1024]  # fit model context
    inputs = tokenizer.encode(short_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs, max_length=180, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def classify_using_similarity(text: str, labels: List[str], embedder) -> str:
    if not text or not labels:
        return "Unknown"

    text_embedding = embedder.encode(text[:512], convert_to_tensor=True)
    label_embeddings = embedder.encode(labels, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(text_embedding, label_embeddings)[0]
    best_idx = torch.argmax(similarity_scores).item()
    return labels[best_idx]

def generate_audio_with_pyttsx3(text: str, path: str):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, path)
        engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"Failed to generate audio: {str(e)}")

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Paper Analyzer", layout="centered")
st.title("🧾 Research Paper Analyzer ")

st.markdown("""
Upload a research paper in PDF format and enter topics you're interested in. This app will:
- Extract content
- Summarize it
- Identify the closest topic using semantic similarity
- Generate an audio summary using `pyttsx3`
""")

with st.form("paper_form"):
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    topic_str = st.text_input("Enter topics separated by commas")
    submit = st.form_submit_button("Analyze")

if submit and pdf_file and topic_str:
    with st.spinner("Processing..."):
        try:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, pdf_file.name)

            with open(file_path, "wb") as f:
                f.write(pdf_file.read())

            # Extract text
            text_content = extract_text_via_pdfplumber(file_path)
            st.info(f"Extracted {len(text_content)} characters.")

            if not text_content.strip():
                st.error("❌ No text extracted.")
            else:
                topics = [t.strip() for t in topic_str.split(",") if t.strip()]
                embedder = load_embedding_model()
                tokenizer, model = load_summarizer()

                summary = summarize_via_model(text_content, tokenizer, model)
                topic = classify_using_similarity(summary, topics, embedder)

                st.markdown(f"### 📌 Detected Topic: `{topic}`")
                st.markdown("### 📄 Summary:")
                st.write(summary)

                audio_path = os.path.join(temp_dir, "summary_audio.mp3")
                generate_audio_with_pyttsx3(summary, audio_path)

                st.markdown("### 🔊 Audio Summary")
                st.audio(audio_path)
                st.success("Complete!")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
