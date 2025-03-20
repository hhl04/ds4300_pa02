from sentence_transformers import SentenceTransformer
import time
import tracemalloc
import numpy as np
from config import EMBEDDING_MODELS  


def get_embedding(text: str, model_name: str) -> list:
    """Generate embeddings using the specified model and ensure the correct dimension."""
    if model_name in EMBEDDING_MODELS:
        model, expected_dim = EMBEDDING_MODELS[model_name] 
        embedding = model.encode(text).tolist()
        
        if len(embedding) != expected_dim:
            raise ValueError(f"Error: Generated vector dimension is {len(embedding)}, expected {expected_dim} for {model_name}.")

        return embedding
    else:
        raise ValueError(f"Model {model_name} is not recognized!")

def benchmark_embedding(text: str, model_name: str):
    """Benchmark embedding generation for speed & memory usage."""
    tracemalloc.start()
    start_time = time.time()

    embedding = get_embedding(text, model_name)

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "model": model_name,
        "vector_dim": len(embedding),  
        "time_sec": elapsed_time,
        "peak_memory_mb": peak / (1024 * 1024)
    }

# FOR TESTING
sample_text = "This is a test sentence for embedding benchmarking."
for model in EMBEDDING_MODELS.keys():
    result = benchmark_embedding(sample_text, model)
    print(f"{result}")  