from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import fitz  # PyMuPDF
import re
import requests
import faiss
from sentence_transformers import SentenceTransformer
from database import init_db, create_session, save_message, get_all_sessions, get_session_messages, register_user, validate_user

app = Flask(__name__)
CORS(app)

print("[📦] Initializing model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("[✓] Model loaded successfully")

vector_index = None
text_chunks = []

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    success, message = register_user(name, email, password)
    return jsonify({"success": success, "message": message})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user_name = validate_user(email, password)
    if user_name:
        print(f"[👤] Logged in: {user_name}")
        return jsonify({"success": True, "message": "Login successful.", "name": user_name})
    else:
        return jsonify({"success": False, "message": "Invalid email or password."})

@app.route('/load_pdf', methods=['POST'])
def load_pdf():
    global vector_index, text_chunks
    data = request.get_json()
    pdf_path = data.get('pdf_path')

    if not pdf_path:
        print("[❌] No pdf_path provided in request.")
        return jsonify({"status": "error", "message": "No pdf_path provided."})

    full_path = os.path.join(os.path.dirname(__file__), pdf_path)
    print(f"[📤] Requested PDF path: {pdf_path}")
    print(f"[🔍] Resolved full path: {full_path}")

    if not os.path.exists(full_path):
        print(f"[❌] File does not exist at path: {full_path}")
        return jsonify({"status": "error", "message": f"PDF not found at {full_path}."})

    print(f"[📄] Loading PDF from {full_path}")
    text = extract_text_from_pdf(full_path)
    text_chunks = chunk_text(text)
    embeddings = embed_chunks(text_chunks)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    vector_index = index

    print("[✅] PDF loaded and embedded successfully.")
    return jsonify({"status": "ok", "message": "PDF loaded."})

@app.route('/create_session', methods=['POST'])
def create_new_session():
    data = request.json
    session_name = data.get('name', 'Untitled Session')
    session_id = create_session(session_name)
    print(f"[💬] New session created: {session_name} (ID: {session_id})")
    return jsonify({"session_id": session_id})

@app.route('/chat', methods=['POST'])
def chat():
    global vector_index, text_chunks
    data = request.json
    question = data.get('message', '')
    session_id = data.get('session_id', None)
    override = data.get('override', False)
    print(f"[→] Received question: {question}")

    if session_id:
        save_message(session_id, "user", question)

    if override:
        print("[🔁] User confirmed fallback to LLM")
        final_answer = rewrite_with_local_llm("", question, mode="generate")
        if session_id:
            save_message(session_id, "bot", final_answer)
        return jsonify({"reply": final_answer, "source": "llm"})

    if vector_index and text_chunks:
        query_vec = embed_query(question).reshape(1, -1)
        D, I = vector_index.search(query_vec, 3)
        chunk = text_chunks[I[0][0]]
        score = get_overlap_score(chunk, question)
        print(f"[📊] Overlap score: {score}")

        if score >= 2:
            print("[🔍] Relevant content matched from PDF")
            sentences = split_into_sentences(chunk)
            ranked = sorted(sentences, key=lambda s: get_overlap_score(s, question), reverse=True)
            top_sentences = [s.strip() for s in ranked if get_overlap_score(s, question) >= 1][:2]
            final_answer = rewrite_with_local_llm(" ".join(top_sentences), question, mode="rewrite")
            if session_id:
                save_message(session_id, "bot", final_answer)
            return jsonify({"reply": final_answer, "source": "pdf"})

        print("[⚠️] No relevant content found in PDF")
        return jsonify({"source": "llm"})

    print("[⚠️] No PDF loaded. Using fallback.")
    final_answer = rewrite_with_local_llm("", question, mode="generate")
    if session_id:
        save_message(session_id, "bot", final_answer)
    return jsonify({"reply": final_answer, "source": "llm"})

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = get_all_sessions()
    result = [{"id": s[0], "name": s[1]} for s in sessions]
    return jsonify({"sessions": result})

@app.route('/get_chat/<int:session_id>', methods=['GET'])
def get_chat(session_id):
    messages = get_session_messages(session_id)
    result = [{"sender": m[0], "message": m[1], "timestamp": m[2]} for m in messages]
    return jsonify({"messages": result})

# === Helpers ===
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=False)

def embed_query(query):
    return model.encode([query])[0]

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def get_overlap_score(text, query):
    return len(set(text.lower().split()) & set(query.lower().split()))

from llama_cpp import Llama

# Initialize LLM once globally
model_path = os.path.join(os.path.dirname(__file__), "models", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)


def rewrite_with_local_llm(content, query, mode="rewrite"):
    if mode == "rewrite":
        prompt = f"""You are an academic assistant. Use the following extracted content from a course PDF to answer the question below in a formal, informative tone. Clearly explain the concept and mention the section if possible.

        Extracted content:
        \"\"\"
        {content}
        \"\"\"

        Question: {query}
        Answer:"""

    else:
        prompt = f"You are a helpful tutor. Answer this question: '{query}'"

    try:
        print("[🧠] Sending prompt to local model:", prompt)
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        print("[📦] Raw LLM response:", result)
        reply = result["choices"][0]["message"]["content"].strip()
        print(f"[←] LLM reply: {reply}")
        return reply
    except Exception as e:
        print(f"[❌] LLM error: {e}")
        return "⚠️ AI response failed. Please try again."

    except Exception as e:
        print(f"[❌] LLM error: {e}")
        return "⚠️ AI response failed. Please try again."

from flask import send_from_directory
import threading
import time
import webbrowser

# Serve static frontend files from root
@app.route('/<path:filename>')
def serve_static_file(filename):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return send_from_directory(root_dir, filename)

# Auto-open browser
def open_browser():
    time.sleep(1)  # Give Flask time to start
    webbrowser.open('http://localhost:5000/home.html')
@app.route('/')
def serve_home():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return send_from_directory(root_dir, 'home.html')

if __name__ == '__main__':
    init_db()
    print("[🚀] Starting Askamedia Flask server...")
    threading.Thread(target=open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)

