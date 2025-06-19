import os
import requests
from tqdm import tqdm

def download_model():
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    # Adjust save_path to save in Models/ directory relative to the current working directory
    save_path = os.path.join(os.getcwd(), "Model", "llama-2-7b-chat.Q4_K_M.gguf")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print("✅ Model already downloaded.")
        return

    print("⬇ Downloading LLaMA 2 GGUF model...")

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(save_path, "wb") as file, tqdm(
        desc=save_path,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print("✅ Download complete!")

if __name__ == "__main__":
    download_model()
