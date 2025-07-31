from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from typing import List
from utils.pdf_loader import extract_text_from_pdf

# 1. Chunking
def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# 2. Local Embedding (FREE)
def get_embeddings(text_chunks: List[str]):
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return embeddings

def get_embedding(query: str):
    return model.encode([query])[0]

# 3. Build FAISS vector store
def build_vector_store(pdf_path: str, save_path='embeddings'):
    print("[+] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("[+] Chunking text...")
    chunks = chunk_text(text)

    print("[+] Generating embeddings from chunks...")
    vectors = get_embeddings(chunks)

    print("[+] Creating FAISS index...")
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, os.path.join(save_path, "index.faiss"))

    with open(os.path.join(save_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("[✓] Vector store built and saved.")

# 4. Load vector store
def load_vector_store(embedding_path='embeddings'):
    index = faiss.read_index(f"{embedding_path}/index.faiss")
    with open(f"{embedding_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# 5. Search vector DB
def search_index(query: str, top_k=1, embedding_path='embeddings'):
    index, chunks = load_vector_store(embedding_path)
    query_vector = get_embedding(query).reshape(1, -1)

    D, I = index.search(query_vector, top_k)
    results = [chunks[i] for i in I[0]]
    return results
