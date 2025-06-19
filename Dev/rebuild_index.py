from pdf_utils import extract_text
from retriever import chunk_text, create_vector_index, save_faiss_index
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
text = extract_text("../Docs/Week_1_merged.pdf")
chunks = chunk_text(text)
index, _ = create_vector_index(chunks, embedder)
save_faiss_index(index, chunks)
print("✅ FAISS index rebuilt and saved successfully.")
