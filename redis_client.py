import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
from process_docs import process_pdfs

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 768
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

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


# def main():
#     clear_redis_store()
#     create_hnsw_index()

#     process_pdfs("/Users/paulchampagne/Desktop/DS 4300/RagIngestAndSearch/data/")
#     print("\n---Done processing PDFs---\n")
#     query_redis("What is the capital of France?")


# if __name__ == "__main__":
#     main()
