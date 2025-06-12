# Welcome to Askamedia 📚🤖

Askamedia is academic chatbot designed to answer questions based solely on Course PDF documents. It is ideal for students, educators, and researchers who want a smart assistant trained on their custom content — without using cloud APIs or pre-trained black-box LLMs.

## 🎯 Key Features

- 📄 PDF-based Q&A: Ask questions from the content of your course.
- ⚙️ Lightweight LLM: Uses a small, efficient fine-tuned model version.
- 🧠 Generates answers using RAG: Retrieval + Lightweight LLM

## Answers are retrieved or generated using only the course content.

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Libraries:** PyPDF2, Transformers, SentencePiece, FAISS
- **Model:** Custom fine-tuned LLM on your uploaded data 
  
Askamedia/
├── Dev/               ← Main chatbot with RAG pipeline (askamedia.py)
├── Model/             ← Lightweight LLM model (custom or fine-tuned)
├── data/              ← Dataset PDFs for Q&A
├── Docx/              ← Course documents
├── requirements.txt   ← All dependencies
├── .gitignore         ← Git ignore rules (e.g., venv/, __pycache__/)
└── README.md          ← Project documentation


---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Jimit177/Askamedia
cd Askamedia

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python Dev/askamedia.py
