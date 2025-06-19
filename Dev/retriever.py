import pickle
import faiss
import numpy as np
def retrieve_top_k(query, k, embedder, index, chunks):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, k)
    top_chunks = [chunks[i] for i in indices[0]]
    top_scores = distances[0]
    return list(zip(top_chunks, top_scores))



def create_vector_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def chunk_text(text, size=300, overlap=50):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) < size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

def save_faiss_index(index, chunks, index_path="../data/vector.index", meta_path="../data/chunks.pkl"):
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_path="../data/vector.index", meta_path="../data/chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
def get_context_from_query(query, embedder, index_path="../data/vector.index", meta_path="../data/chunks.pkl", top_k=2):
    index, chunk_list = load_faiss_index(index_path, meta_path)
    results = retrieve_top_k(query, top_k, embedder, index, chunk_list)
    return results  # Now returns list of (chunk, score)


