from sentence_transformers import SentenceTransformer
import ollama
import time
import tracemalloc
import numpy as np
from config import EMBEDDING_MODELS


def get_embedding(text: str, model_name="all-MiniLM-L6-v2") -> list:
    """Get embedding for a chunk of text using specified model."""
    if model_name == "ollama-nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif model_name in EMBEDDING_MODELS:
        model = EMBEDDING_MODELS[model_name]
        embedding = model.encode(text)
        return embedding.tolist()
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
def benchmark_embedding(text: str, model_name: str):
    """Benchmark embedding generation for speed & memory usage."""
    tracemalloc.start()
    start_time = time.time()

    _ = get_embedding(text, model_name)

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "model": model_name,
        "time_sec": elapsed_time,
        "peak_memory_mb": peak / (1024 * 1024)
    }

# FOR TESTING
# sample_text = "This is a test sentence for embedding benchmarking."
# for model in EMBEDDING_MODELS.keys():
#     #result = benchmark_embedding(sample_text, model)
#     print(model)
#     r = get_embedding(sample_text, model)
#     print(r)
#     print()
#     #print(f"{result}")