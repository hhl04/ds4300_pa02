import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
from process_docs import process_pdfs
from embeddings import get_embedding, benchmark_embedding
from config import EMBEDDING_MODELS

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

# Define each vector dim
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,          
    "all-mpnet-base-v2": 768,         
    "InstructorXL": 768,              
    "ollama-nomic": 1536,             
}

#VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


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

    schema = "text TEXT"
    for model_name, dim in MODEL_DIMS.items():
        schema += f" embedding_{model_name} VECTOR HNSW 6 DIM {dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}"

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA {schema}
        """
    )
    print("Index created successfully.")

def sanitize_model_name(model_name: str):
    return model_name.replace("-", "_")

# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embeddings: dict):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    mapping = {
        "file": file,
        "page": page,
        "chunk": chunk,
        #"text": chunk,  # Added 'text' to satisfy index schema
    }
    for model_name, embedding in embeddings.items():
        field_name = f"embedding_{sanitize_model_name(model_name)}"
        mapping[field_name] = np.array(
            embedding, dtype=np.float32
        ).tobytes()

    redis_client.hset(key, mapping=mapping)
    print(f"Stored embeddings for: {chunk}")

def run_ingestion(data_dir):
    chunks = process_pdfs(data_dir)
    for chunk_data in chunks:
        chunk = chunk_data["chunk"]
        file = chunk_data["file"]
        page = chunk_data["page"]
        chunk_index = chunk_data["chunk_index"]
        chunk_size = chunk_data["chunk_size"]
        overlap = chunk_data["overlap"]

        embeddings = {}
        for model_name in EMBEDDING_MODELS.keys():
            embedding = get_embedding(chunk, model_name)
            embeddings[model_name] = embedding

        store_embedding(
            file=file,
            page=str(page),
            chunk=f"chunk_{chunk_index}_cs{chunk_size}_ov{overlap}",
            embeddings=embeddings,
        )

def query_redis(query_text: str, model_name: str):
    field = f"embedding_{sanitize_model_name(model_name)}"

    q = (
        Query(f"*=>[KNN 5 @{field} $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance AS vector_distance")
        .dialect(2)
    )

    embedding = get_embedding(query_text, model_name)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    for doc in res.docs:
        print(f"{doc.id} | {getattr(doc, 'vector_distance', 'N/A')} | Chunk: {doc.chunk}")



# TESTING
if __name__ == "__main__":
    # clear_redis_store()
    # create_hnsw_index()
    # run_ingestion("/Users/paulchampagne/Desktop/DS 4300/ds4300_pa02/ds4300 docs/")
    # print("✅ Ingestion completed!")

    # Optionally test a query right away
    query_redis("MongoDB is Cool!",model_name="all-mpnet-base-v2")
