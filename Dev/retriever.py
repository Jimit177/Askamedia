import pickle
import faiss
import numpy as np
def retrieve_top_k(query, k, embedder, index, chunks):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def create_vector_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

def save_faiss_index(index, chunks, index_path="data/vector.index", meta_path="data/chunks.pkl"):
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_path="data/vector.index", meta_path="data/chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
