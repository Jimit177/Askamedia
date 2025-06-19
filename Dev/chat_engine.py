from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from retriever import get_context_from_query
import streamlit as st
import os

def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    model_path = "../Model/llama-2-7b-chat.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ LLaMA GGUF model not found at {model_path}.")
    qa_model = Llama(model_path=model_path, n_ctx=2048, n_threads=8, n_gpu_layers=0)
    return embedder, qa_model

def build_prompt(chat_history, context, question):
    return (
        f"[INST] You are a helpful AI tutor. ONLY use the course material below to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"If the answer is not found in the course material above, you must say: "
        f"'I’m sorry, this question is outside the course scope.' "
        f"Do not guess. Do not invent answers. Do not use external sources. Never fabricate links. [/INST]"
    )

def generate_answer(prompt, qa_model):
    output_placeholder = st.empty()
    output = ""
    for chunk in qa_model(prompt, max_tokens=512, stream=True, stop=["</s>"]):
        token = chunk["choices"][0]["text"]
        output += token
        output_placeholder.markdown(f"**🤖 Askamedia:** {output}▌")
    output_placeholder.markdown(f"**🤖 Askamedia:** {output}")
    return output.strip()

def get_response(query, embedder, qa_model, score_threshold=0.75):
    retrieved = get_context_from_query(query, embedder)
    if not retrieved:
        return "⚠️ I’m sorry, I couldn’t find anything in the course materials related to that question."
    top_chunk, top_score = retrieved[0]
    if top_score < score_threshold:
        return "⚠️ This question seems unrelated to the course content."
    prompt = build_prompt([], top_chunk, query)
    return generate_answer(prompt, qa_model)
