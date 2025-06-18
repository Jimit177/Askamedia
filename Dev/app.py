import os
import random
import time
import streamlit as st
from pdf_utils import extract_text
from retriever import chunk_text, create_vector_index, retrieve_top_k
from retriever import save_faiss_index, load_faiss_index
from chat_engine import build_prompt, generate_answer, load_models

# ──────────────────────────────────────────────
# UI CONFIG
st.set_page_config(page_title="Askamedia - Course Chatbot", layout="centered")

# ──────────────────────────────────────────────
# SESSION CONTROL
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ──────────────────────────────────────────────
# LOGIN PAGE FUNCTION
def login_page():
    st.image("../Docs/logo.png", width=120)

    st.markdown("<h2 style='text-align: center;'>Welcome to Askamedia</h2>", unsafe_allow_html=True)

    catchphrases = [
        "Your questions, answered instantly.",
        "Smarter learning starts here.",
        "Study support at your fingertips.",
        "Ask. Learn. Succeed.",
        "The AI tutor that never sleeps.",
        "Get help. Anytime. Anywhere.",
        "One login away from clarity.",
        "Your learning companion, 24/7."
    ]

    phrase_placeholder = st.empty()
    for _ in range(3):
        phrase = random.choice(catchphrases)
        phrase_placeholder.markdown(
            f"<h4 style='text-align: center; color: gray;'>{phrase}</h4>",
            unsafe_allow_html=True
        )
        time.sleep(1)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.warning("Please enter both fields.")

# ──────────────────────────────────────────────
# MAIN CHATBOT FUNCTION
def chatbot_page():
    st.title("📘 Askamedia - Course Chatbot")
    st.info("Ask any question based on the course material.")

    # Paths
    pdf_path = os.path.join("../Docs", "Week_1_merged.pdf")
    index_path = "../data/vector.index"
    meta_path = "../data/chunks.pkl"

    # Load models
    embedder, qa_model = load_models()

    # Load or build index
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

    # Input from user
    with st.chat_message("user"):
        query = st.text_input("❓ Ask something about the course")

    if query:
        # Encode query and get top-k matches + distances
        query_vector = embedder.encode([query])
        distances, indices = index.search(query_vector, 3)

        if distances[0][0] > 1.2:
            answer = "⚠️ This doesn't seem related to the course material. Please try asking something from the PDF content.We will be implementing fallback policy in future"
        else:
            relevant_chunks = [chunk_list[i] for i in indices[0]]
            context = "\n".join(relevant_chunks)
            prompt = build_prompt(st.session_state.chat_history, context, query)
            answer = generate_answer(prompt, qa_model)

        
            

        st.session_state.chat_history.append((query, answer))

        # Display answer
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Chat History
    with st.expander("🕓 Chat History"):
        for q, a in st.session_state.chat_history:
            st.markdown(f"<b style='color:blue'>Q:</b> {q}", unsafe_allow_html=True)
            st.markdown(f"<b style='color:green'>A:</b> {a}", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ROUTING
if not st.session_state.logged_in:
    login_page()
else:
    chatbot_page()
