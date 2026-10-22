
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-local"

payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Explain RAG in three simple sentences."
        }
    ],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0.2
    }
}

try:
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )
    response.raise_for_status()

    data = response.json()
    print("\n=== QWEN3 RESPONSE ===")
    print(data["message"]["content"])

except requests.exceptions.RequestException as error:
    print("Could not connect to Ollama:", error)
