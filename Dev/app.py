import os
import streamlit as st
from pdf_utils import extract_text
from retriever import chunk_text, create_vector_index, retrieve_top_k
from retriever import save_faiss_index, load_faiss_index
from chat_engine import build_prompt, generate_answer, load_models

# ──────────────────────────────────────────────
# 📘 UI CONFIG
st.set_page_config(page_title="Askamedia - Course Chatbot")
st.title("📘 Askamedia - Course Chatbot")
st.info("Ask any question based on the course material.")

# ──────────────────────────────────────────────
# 📄 PATHS

pdf_path = os.path.join("../Docs", "Week_1_merged.pdf")
index_path = "../data/vector.index"
meta_path = "../data/chunks.pkl"

# ──────────────────────────────────────────────
# 🧠 LOAD MODELS
embedder, qa_model = load_models()

# ──────────────────────────────────────────────
# 🔄 LOAD OR BUILD FAISS INDEX
if os.path.exists(index_path) and os.path.exists(meta_path):
    index, chunk_list = load_faiss_index(index_path, meta_path)
else:
    text = extract_text(pdf_path)
    if not text.strip():
        st.error("No text could be extracted from the PDF.")
        st.stop()
    chunk_list = chunk_text(text)
    index, chunk_list = create_vector_index(chunk_list, embedder)
    save_faiss_index(index, chunk_list, index_path, meta_path)

# ──────────────────────────────────────────────
# 💬 CHAT SESSION MEMORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ──────────────────────────────────────────────
# ❓ USER QUESTION
query = st.text_input("❓ Ask a question")

if query:
    # 🔍 Define fallback questions
    fallback_questions = [
    "what is quantum computing",
    "who is the prime minister of india",
    "how do rockets work",
    "who won the world cup",
    "explain photosynthesis",
    "what is the capital of france", 
    "where can i get a pizza",
    "what is nano technology",
    "what is my name"
]


    normalized_query = query.strip().lower()

    if any(normalized_query.startswith(q) for q in fallback_questions):
        answer = "🛠️ We're working on a fallback policy for this topic. Please ask something related to the course material."
    else:
        # 🔎 RETRIEVE TOP K CHUNKS
        relevant_chunks = retrieve_top_k(query, 3, embedder, index, chunk_list)
        context = "\n".join(relevant_chunks)

        # 🧠 BUILD PROMPT & GENERATE ANSWER
        prompt = build_prompt(st.session_state.chat_history, context, query)
        answer = generate_answer(prompt, qa_model)


    # 📜 DISPLAY ANSWER
    st.markdown(f"**🧠 Answer:** {answer}")

    # 💾 UPDATE CHAT MEMORY
    st.session_state.chat_history.append((query, answer))

# 🕓 SHOW CHAT HISTORY
with st.expander("🕓 Chat History"):
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
