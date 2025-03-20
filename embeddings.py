from sentence_transformers import SentenceTransformer
import time
import tracemalloc
import numpy as np
from config import EMBEDDING_MODELS

# Only use 768-dimensional embedding models for now
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),  
}

DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

def get_embedding(text: str, model_name=DEFAULT_EMBEDDING_MODEL) -> list:
    """Ensure that the returned vector is 768-dimensional"""
    if model_name in EMBEDDING_MODELS:
        model = EMBEDDING_MODELS[model_name]
        embedding = model.encode(text)
        embedding = embedding.tolist()
        
        if len(embedding) != 768:
            raise ValueError(f"Error: Generated vector dimension is {len(embedding)}")

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

