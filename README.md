# 🧠 Research Paper Summarization & Classification System

This project is a Streamlit-powered application that allows users to upload research paper PDFs, generate text summaries, classify them into user-specified topics, and even generate audio summaries with the help of Text-to-Speech.

---

## 🎬 Demonstration

Watch a quick walkthrough of how to use the app on Hugging Face Spaces:

[![Demonstration Video](https://img.youtube.com/vi/placeholder/0.jpg)](https://github.com/user-attachments/assets/31be46ab-c22f-4e41-af2d-61141bf283fd)

---

## 🔧 Setup Instructions

### ▶️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dhiraut/Research_Paper_Summarization_Multi_Agent_System.git
   cd Research-Paper-Analyzer
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   Here's a snippet from `requirements.txt`:
   streamlit
   PyMuPDF
   transformers
   torch
   sentence-transformers
   gtts
   ```

4. **Run the app:**
   ```bash
   streamlit run main.py
   ```

## 🏗️ System Overview

---

This monolithic app (app.py) mimics a multi-agent system through modular logic. Each function represents an agent role:

| Agent                      | Role                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| 📄 PDF Text Extractor     | Uses PyMuPDF to extract clean text from uploaded PDFs               |
| 🧠 Topic Classifier Agent   | Classifies paper content using `zero-shot-classification`.               |
| ✍️ Summarizer Agent         | Uses transformer models to generate concise summaries.                   |
| 🎙️ Audio Generation Agent  | Converts summary text into audio with `gTTS`.                            |

Each logical module acts independently and communicates via function calls, allowing future conversion into fully decoupled microservices or agent-based components.

---

## 🧠 How It Works

1. **Upload** a research paper (PDF).
2. **Enter** one or more topics for classification.
3. The app:
   - 📄 Extracts text from the paper
   - ✍️ Summarizes the content using transformer models
   - 🧠 Classifies the document to the closest topic
   - 🔊 Generates a downloadable audio version of the summary
4. 🖥️ All results are displayed in an intuitive **Streamlit UI**.

---

## 🧪 Key Technologies Used

- 🧠 **Transformers** for summarization [`sshleifer/distilbart-cnn-12-6`]
- 💡 **Sentence Embeddings** for classification [`sentence-transformers/all-MiniLM-L6-v2`]
- 📄 **PyMuPDF** for PDF parsing
- 🔊 **gTTS** for text-to-speech audio generation
- ⚡ **Streamlit** for UI

---

## ⚠️ Limitations and Future Improvements

## ⚠️ Limitations

- 📄 Only one paper is processed at a time.
- 💾 No local storage for summaries/audio files.
- 🌐 `gTTS` requires an internet connection and does **not** support offline mode.
- 🐢 Large PDFs may result in slower processing or truncation during summarization.

## 🚀 Future Enhancements

- 📚 Add support for batch summarization or cross-paper analysis
- 🔄 Replace `gTTS` with offline TTS options like `pyttsx3` or `Coqui`
- 🌐 Enable DOI or URL-based input from arXiv/Semantic Scholar
- 🧩 Modularize agents into separate microservices or FastAPI endpoints
- 🧠 Add vector-based search and archiving for efficient information retrieval

---

## 💡 Tech Stack

- Python 3.10+
- Streamlit
- Hugging Face Transformers
- PyMuPDF
- gTTS

---

## 🧠 Author Notes

This project showcases how NLP and LLMs can accelerate research understanding and classification. It's designed for learners, researchers, and developers seeking insights from dense academic material.

## 👩‍💻 Developed by
Bhavya Sri Kandru
Computer Science | Data Analytics Enthusiast
