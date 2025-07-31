import requests

# Construct your prompt to test the local model
prompt = "Explain reinforcement learning in simple terms."

# Send request to LM Studio's local OpenAI-style API
response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "local-model",  # LM Studio ignores this name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7  # optional creativity control
    }
)

# Print out the result
if response.status_code == 200:
    print("✅ Local LLM Response:")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"❌ Error {response.status_code}: {response.text}")
