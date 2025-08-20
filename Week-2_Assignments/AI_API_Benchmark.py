import time
import requests
from dotenv import load_dotenv
import os
import statistics

# -----------------------------
# Load API keys from .env
# -----------------------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY") 

HF_MODEL = "facebook/bart-large-cnn"
OLLAMA_MODEL = "gemma:2b"

PROMPTS = [
    "Who is Edward Snowden?",
    "Summarize: New trump tarif rule and regulations.",
    "Classify sentiment (Positive/Negative/Neutral): I love iphone!"
]


# --- Clients ---
def hf_call(prompt):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 64}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()
    # The result is usually a list of dicts with 'summary_text'
    if isinstance(result, list) and "summary_text" in result[0]:
        return result[0]["summary_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    return str(result)


def ollama_call(prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,  # Use the correct model name from variable
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    result = r.json()
    # Ollama response structure may vary; check for 'message' and 'content'
    if "message" in result and "content" in result["message"]:
        return result["message"]["content"]
    elif "response" in result:
        return result["response"]
    return str(result)


# --- Benchmark ---
def benchmark(name, fn):
    times = []
    for p in PROMPTS:
        t0 = time.perf_counter()
        try:
            out = fn(p)
        except Exception as e:
            out = f"Error: {e}"
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[{name}] Prompt: {p}\n → {out.strip()}\n ({dt:.2f}s)\n")
    print(f"== {name} Summary ==")
    print(f"mean: {statistics.mean(times):.2f}s | p95: {sorted(times)[int(0.95*len(times))-1]:.2f}s | min: {min(times):.2f}s | max: {max(times):.2f}s\n")


if __name__ == "__main__":
    print("Benchmarking Hugging Face Router...")
    benchmark("HuggingFace", hf_call)

    print("Benchmarking Ollama...")
    benchmark("Ollama", ollama_call)

   