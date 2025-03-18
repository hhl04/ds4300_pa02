from sentence_transformers import SentenceTransformer

CHUNK_SIZES = [200, 500, 1000]
OVERLAPS = [0, 50, 100]
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "InstructorXL": SentenceTransformer("hkunlp/instructor-xl"),
    "ollama-nomic": "nomic-embed-text",
}
REDIS_HOST = "localhost"
REDIS_PORT = 6380
