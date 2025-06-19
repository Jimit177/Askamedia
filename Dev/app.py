import streamlit as st
from chat_engine import load_models, get_response

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Askamedia", page_icon="📘", layout="wide")

# ---------------------------
# Session State Initialization
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedder" not in st.session_state or "qa_model" not in st.session_state:
    with st.spinner("🔄 Loading models..."):
        embedder, qa_model = load_models()
        st.session_state.embedder = embedder
        st.session_state.qa_model = qa_model

# ---------------------------
# Page 1: Login
# ---------------------------
if st.session_state.page == "login":
    st.title("🔐 Askamedia Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email and password:  # Dummy check
            st.session_state.logged_in = True
            st.session_state.page = "course"
        else:
            st.warning("Please enter both email and password.")

# ---------------------------
# Page 2: Course Selection
# ---------------------------
elif st.session_state.page == "course":
    st.title("📘 Select Your Course")
    course = st.selectbox("Choose a course to continue:", ["Artificial Intelligence,Algorithms and Mathematics"])
    if st.button("Continue"):
        st.session_state.selected_course = course
        st.session_state.page = "chat"

# ---------------------------
# Page 3: Chatbot
# ---------------------------
elif st.session_state.page == "chat":
    st.markdown(f"<h2 style='text-align: center;'>🤖 Askamedia – {st.session_state.selected_course}</h2>", unsafe_allow_html=True)
    query = st.text_input("Ask your question here:")
    if query:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = get_response(query, st.session_state.embedder, st.session_state.qa_model, score_threshold=0.75)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
        st.session_state.chat_history.append({"user": query, "bot": answer})

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑‍🎓 You:** {chat['user']}")
        st.markdown(f"**🤖 Askamedia:** {chat['bot']}")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

    if st.button("🔙 Back to Course Selection"):
        st.session_state.page = "course"
