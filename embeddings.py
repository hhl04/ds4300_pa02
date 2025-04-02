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

  