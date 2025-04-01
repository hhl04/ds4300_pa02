
import redis
import numpy as np
import ollama
from redis.commands.search.query import Query
from redis_process_docs import process_pdfs
from embeddings import get_embedding
from config import EMBEDDING_MODELS
from tqdm import tqdm

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def sanitize_model_name(model_name):
    return model_name.replace("-", "_")

def get_vector_field_name(model_name):
    return f"embedding_{sanitize_model_name(model_name)}"

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    schema = "text TEXT chunk_size NUMERIC overlap NUMERIC"
    for model_name, (_, dim) in EMBEDDING_MODELS.items():
        field_name = get_vector_field_name(model_name)
        schema += f" {field_name} VECTOR HNSW 6 DIM {dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}"

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA {schema}
        """
    )
    print("Index created successfully.")



# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, chunk_size: int, overlap: int, embeddings: dict):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    mapping = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    for model_name, embedding in embeddings.items():
        field_name = get_vector_field_name(model_name)
        mapping[field_name] = np.array(
            embedding, dtype=np.float32
        ).tobytes()

    redis_client.hset(key, mapping=mapping)

def run_ingestion(data_dir):
    chunks = process_pdfs(data_dir)
    for chunk_data in tqdm(chunks, desc="Storing chunks in Redis"):
        chunk = chunk_data["chunk"]
        file = chunk_data["file"]
        page = chunk_data["page"]
        chunk_size = chunk_data["chunk_size"]
        overlap = chunk_data["overlap"]

        embeddings = {}
        for model_name, (model, _) in EMBEDDING_MODELS.items():
            embedding = get_embedding(chunk, model_name=model_name, model=model)
            embeddings[model_name] = embedding

        store_embedding(
            file=file,
            page=str(page),
            chunk=f"file_{file}_cs{chunk_size}_ov{overlap}",
            chunk_size=chunk_size,
            overlap=overlap,
            embeddings=embeddings,
        )

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

def query_redis(query_text: str, model_name: str):
    field = get_vector_field_name(model_name)
    model, _ = EMBEDDING_MODELS[model_name]

    q = (
        Query(f"*=>[KNN 5 @{field} $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )

    embedding = get_embedding(query_text, model_name=model_name, model=model)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    if not res.docs:
        print("No matches found for your query.")
        return

    print(f"Found {len(res.docs)} result(s):")
    for doc in res.docs:
        print(f"{doc.id} | Distance: {doc.vector_distance} | File: {doc.file} | Chunk: {doc.chunk}")

# TESTING
if __name__ == "__main__":
    clear_redis_store()
    create_hnsw_index()
    run_ingestion("/Users/paulchampagne/Desktop/DS 4300/ds4300_pa02/ds4300_testdocs/")
    print("Ingestion completed!")
