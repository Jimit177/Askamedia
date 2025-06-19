import streamlit as st
from chat_engine import load_models, get_response
from retriever import load_faiss_index
from pdf_utils import extract_text
import os

# ---------------------------
# Set Page Config
# ---------------------------
st.set_page_config(page_title="Askamedia", page_icon="📘", layout="wide")

# ---------------------------
# Initialize session state
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Load models
# ---------------------------
if "embedder" not in st.session_state or "qa_model" not in st.session_state:
    with st.spinner("🔄 Loading models..."):
        embedder, qa_model = load_models()
        st.session_state.embedder = embedder
        st.session_state.qa_model = qa_model
else:
    embedder = st.session_state.embedder
    qa_model = st.session_state.qa_model

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("../Docs/logo.png", width=180)
    st.title("Askamedia 📘")
    st.markdown("Smarter learning starts here. Ask anything from your course PDF.")

    uploaded_file = st.file_uploader("Upload a new PDF (optional)", type=["pdf"])
    if uploaded_file:
        with open("Docs/Week_1_merged.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ PDF uploaded. Please re-run the FAISS index build script.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# ---------------------------
# Main Chat Interface
# ---------------------------
st.markdown("<h2 style='text-align: center;'>Askamedia: Your Course Companion 🤖</h2>", unsafe_allow_html=True)

query = st.text_input("Ask your question here:")

if query:
    with st.spinner("🤔 Thinking..."):
        try:
            answer = get_response(query, embedder, qa_model, score_threshold=0.75)
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

    st.session_state.chat_history.append({"user": query, "bot": answer})

# ---------------------------
# Chat History Display
# ---------------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑‍🎓 You:** {chat['user']}")
    st.markdown(f"**🤖 Askamedia:** {chat['bot']}")
