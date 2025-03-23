from sentence_transformers import SentenceTransformer
import ollama
from config import EMBEDDING_MODELS

# Initialize models
models = {}
for model_name in EMBEDDING_MODELS.keys():
    if model_name == "nomic-embed-text":
        # Ollama model doesn't need pre-loading
        models[model_name] = None
    else:
        try:
            models[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            models[model_name] = None

def get_embedding(text: str, model_name: str) -> list:
    """Get embedding vector for text"""
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")
    
    if model_name == "nomic-embed-text":
        # Use Ollama for embedding
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error getting embedding from Ollama: {e}")
            return None
    else:
        # Use SentenceTransformer for embedding
        try:
            model = models[model_name]
            if model is None:
                raise ValueError(f"Model {model_name} not properly initialized")
            return model.encode(text).tolist()
        except Exception as e:
            print(f"Error getting embedding from {model_name}: {e}")
            return None

def benchmark_embedding(text: str, model_name: str) -> dict:
    """Benchmark embedding model performance"""
    import time
    
    start_time = time.time()
    embedding = get_embedding(text, model_name)
    end_time = time.time()
    
    if embedding is None:
        return {
            "success": False,
            "error": f"Failed to get embedding from {model_name}"
        }
    
    return {
        "success": True,
        "time_taken": end_time - start_time,
        "embedding_dimension": len(embedding)
    }

if __name__ == "__main__":
    # Test code
    sample_text = "This is a test sentence for embedding benchmarking."
    
    for model_name in EMBEDDING_MODELS.keys():
        print(f"\nTesting model: {model_name}")
        result = benchmark_embedding(sample_text, model_name)
        print(f"Result: {result}")  