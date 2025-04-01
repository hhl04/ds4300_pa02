from sentence_transformers import SentenceTransformer
import time
import tracemalloc
import numpy as np
from config import EMBEDDING_MODELS
import ollama
    
def get_embedding(text: str, model_name=None, model=None) -> list:
    """Get embedding for a chunk of text using specified model."""
    text = str(text)

    if model_name == "ollama-nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif model:
        embedding = model.encode(text)
        return embedding.tolist()
    else:
        raise ValueError(f"Model {model_name} not recognized or not passed.")

# def benchmark_embedding(text: str, model_name: str, model) -> dict:
#     import time
#     import tracemalloc

#     tracemalloc.start()
#     start = time.time()
#     embedding = get_embedding(text, model_name=model_name, model=model)
#     end = time.time()
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()

#     return {
#         "model": model_name,
#         "vector_dim": len(embedding),
#         "time_sec": end - start,
#         "peak_memory_mb": peak / 1024 / 1024,
#     }

# FOR TESTING
# if __name__ == "__main__":
#     from config import EMBEDDING_MODELS

#     sample_text = "MongoDB is a document database"
#     for model_name, (model, _) in EMBEDDING_MODELS.items():
#         result = benchmark_embedding(sample_text, model_name=model_name, model=model)
#         print(result)
  