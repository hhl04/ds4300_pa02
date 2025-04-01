from sentence_transformers import SentenceTransformer

CHUNK_SIZES = [200, 500, 1000]
OVERLAPS = [0, 50, 100]

# # Selected three embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": (SentenceTransformer("all-MiniLM-L6-v2"), 384),  
    "all-mpnet-base-v2": (SentenceTransformer("all-mpnet-base-v2"), 768),  
    "InstructorXL": (SentenceTransformer("hkunlp/instructor-xl"), 768),
    "ollama-nomic": ("nomic-embed-text", 768) 
}

# EMBEDDING_MODELS = {
#     "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
#     "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
#     "InstructorXL": SentenceTransformer("hkunlp/instructor-xl"),
#     "ollama-nomic": "nomic-embed-text",
# }

VECTOR_INDEXES = {
    "all-MiniLM-L6-v2": "embedding_index_384",
    "all-mpnet-base-v2": "embedding_index_768",
    "InstructorXL": "embedding_index_768",
}

REDIS_HOST = "localhost"
REDIS_PORT = 6379


