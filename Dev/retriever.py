# retriever.py

import faiss
import numpy as np

# === SECTION 2: Embedding & Vector Indexing ===
def create_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_context(question, chunks, embedder, index, top_k=3):
    q_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)
    return "\n".join([chunks[i] for i in indices[0]])