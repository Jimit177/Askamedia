
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from reader import load_all_pdfs_from_folder
from retriever import create_faiss_index, retrieve_context

# Load local LLaMA model
llm = Llama(
    model_path="Model/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=4,
    verbose=False
)

# === Strict RAG Answer Generation ===
def ask_llm(context, question):
    if not context.strip() or context.strip().lower() in ["", "none", "null"]:
        return "We will be adding fallback support in the future. For now, Askamedia only answers based on the uploaded content."

    prompt = f"""[INST] Use the following context to answer the question.

Context:
{context}

Question: {question}

If the answer is not exactly found in the context, reply with exactly:
"We will be adding fallback support in the future. For now, Askamedia only answers based on the uploaded content."

Answer: [/INST]"""

    output = llm(prompt, max_tokens=1024, stop=["</s>"])
    return output["choices"][0]["text"].strip()


# === Launch Askamedia ===
if __name__ == "__main__":
    print("Launching Askamedia — your local AI assistant 📚🤖")
    chunks = load_all_pdfs_from_folder("data")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = create_faiss_index(chunks, embedder)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye from Askamedia!")
                break
            context = retrieve_context(user_input, chunks, embedder, index)
            response = ask_llm(context, user_input)
            print("Askamedia:", response)
        except KeyboardInterrupt:
            print("\nSession ended.")
            break