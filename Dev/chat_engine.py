from transformers import pipeline

def load_models():
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ✅ Load your locally fine-tuned model
    qa_model = pipeline(
    "text2text-generation",
    model="../Model/flan_t5_finetuned_model",
    tokenizer="../Model/flan_t5_finetuned_model",


    )
    return embedder, qa_model


def build_prompt(chat_history, context, question):
    return (
        f"You are a helpful tutor. Use only the provided course context to answer the student's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer in clear, natural language suitable for students."
    )


def generate_answer(prompt, qa_model):
    result = qa_model(prompt, max_new_tokens=200)[0]["generated_text"]
    return result.strip()
