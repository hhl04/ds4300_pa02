import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
from embeddings import get_embedding
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS
from preprocessing import extract_clean_pdf, split_text_variants
from tqdm import tqdm

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def sanitize_model_name(model_name):
    return model_name.replace("-", "_")

def get_vector_field_name(model_name):
    return "embedding"  # consistent with ingestion logic

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index for each model
def create_index_for_model(model_name, dim):
    index_name = f"embedding_index_{sanitize_model_name(model_name)}"
    try:
        redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
    except redis.exceptions.ResponseError:
        pass

    schema = (
        "file TEXT page TEXT chunk TEXT chunk_size NUMERIC overlap NUMERIC "
        f"{get_vector_field_name(model_name)} VECTOR HNSW 6 DIM {dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}"
    )

    redis_client.execute_command(
        f"""
        FT.CREATE {index_name} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA {schema}
        """
    )
    print(f"Index created successfully for {model_name}!")

# store the embedding in Redis for a specific model
def store_embedding(file: str, page: str, chunk: str, chunk_size: int, overlap: int, model_name: str, embedding):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}_{model_name}"
    field_name = get_vector_field_name(model_name)
    mapping = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "chunk_size": chunk_size,
        "overlap": overlap,
        field_name: np.array(embedding, dtype=np.float32).tobytes()
    }
    redis_client.hset(key, mapping=mapping)


def run_ingestion(data_dir, chunk_size=300, overlap=50):
    for model_name, (model, dim) in EMBEDDING_MODELS.items():
        create_index_for_model(model_name, dim)
        all_chunks = []

        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".pdf"):
                continue

            pdf_path = os.path.join(data_dir, file_name)
            extracted_pages = extract_clean_pdf(pdf_path)

            for page_num, text in extracted_pages.items():
                chunks = split_text_variants(
                    text,
                    file_name=file_name,
                    page_num=page_num,
                    chunk_sizes=[chunk_size],
                    overlaps=[overlap]
                )
                all_chunks.extend(chunks)

        # TQDM now wraps storing phase only
        for chunk_data in tqdm(all_chunks, desc=f"🔁 Ingesting into Redis ({model_name})"):
            chunk = chunk_data["chunk"]
            chunk_size = chunk_data["chunk_size"]
            overlap = chunk_data["overlap"]
            file = chunk_data["file"]
            page = chunk_data["page"]

            embedding = get_embedding(chunk, model_name=model_name, model=model)

            store_embedding(
                file=file,
                page=str(page),
                chunk=f"file_{file}_cs{chunk_size}_ov{overlap}",
                chunk_size=chunk_size,
                overlap=overlap,
                model_name=model_name,
                embedding=embedding,
            )

        print(f"✅ Finished ingestion for model: {model_name}\n")


def query_redis(query_text: str, model_name: str):
    field = get_vector_field_name(model_name)
    index_name = f"embedding_index_{sanitize_model_name(model_name)}"
    model, _ = EMBEDDING_MODELS[model_name]

    q = (
        Query(f"*=>[KNN 5 @{field} $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )

    embedding = get_embedding(query_text, model_name=model_name, model=model)
    res = redis_client.ft(index_name).search(
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
    run_ingestion("/Users/paulchampagne/Desktop/DS 4300/ds4300_pa02/ds4300 docs/")
    print("Ingestion completed!")