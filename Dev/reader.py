# reader.py

import fitz  # PyMuPDF
import os

# === Load and split all PDFs from data/ folder ===
def load_all_pdfs_from_folder(folder_path="data", chunk_size=300):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            doc = fitz.open(full_path)
            for page in doc:
                text = page.get_text()
                for i in range(0, len(text), chunk_size):
                    chunks.append(text[i:i+chunk_size])
    return chunks