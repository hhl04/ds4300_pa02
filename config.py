from sentence_transformers import SentenceTransformer
import ollama

# Embedding model configurations
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": (SentenceTransformer("all-MiniLM-L6-v2"), 384),
    "all-mpnet-base-v2": (SentenceTransformer("all-mpnet-base-v2"), 768),
    "nomic-embed-text": (None, 768)  # Ollama model
}

# Chunking strategies
CHUNK_SIZES = [500, 1000]
CHUNK_OVERLAPS = [50, 100]  

# Vector database configurations
VECTOR_DBS = {
    "redis": {
        "host": "localhost",
        "port": 6379,
        "index_prefix": "embedding_index_"
    },
    "milvus": {
        "host": "localhost",
        "port": 19530
    },
    "chromadb": {
        "host": "localhost",
        "port": 8000
    }
}

# LLM model configurations
LLM_MODELS = {
    "mistral": "mistral:latest",
    "llama2": "llama2:latest"
}

# Test queries
TEST_QUERIES = [
    "What is MongoDB?",
    "How does Redis work?",
    "What are the differences between SQL and NoSQL?",
    "Explain vector databases",
    "What is document storage?"
]

VECTOR_INDEXES = {
    "all-MiniLM-L6-v2": "embedding_index_384",
    "all-mpnet-base-v2": "embedding_index_768",
    "InstructorXL": "embedding_index_768",
}

REDIS_HOST = "localhost"
REDIS_PORT = 6379


