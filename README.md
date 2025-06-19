# Welcome to Askamedia 📚🤖

Askamedia is academic chatbot designed to answer questions based solely on Course PDF documents. It is ideal for students, educators, and researchers who want a smart assistant trained on their custom content — without using cloud APIs or pre-trained black-box LLMs.

## Group members:
1.	Ankit Soni  
2.	Dhruv Chaudhary 
3.	Jay Sindhal 
4.	Jimit Jain 
5.	Vishal Shah

## 🎯 Key Features

- 📄 PDF-based Q&A: Ask questions from the content of your course.
- ⚙️ Lightweight LLM: Uses a small, efficient fine-tuned model version.
- 🧠 Generates answers using RAG: Retrieval + Lightweight LLM

## Answers are retrieved or generated using only the course content.

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Libraries:** PyPDF2, Transformers, SentencePiece, FAISS
- **Model:** fine-tuned LLM
  
## 📁 Project Structure

```
Askamedia/
├── Dev/               ← Main chatbot with RAG pipeline
├── Model/             ← Lightweight LLM model 
├── data/              ← Dataset PDFs for Q&A
├── Docx/              ← Course document
├── requirements.txt   ← All dependencies
├── .gitignore         ← Git ignore rules 
└── README.md          ← Project documentation
```



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
