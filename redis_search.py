#import redis_client  # assuming this has the redis schema setup etc.
import redis_test as redis_client
import numpy as np
from embeddings import get_embedding
import ollama
from redis.commands.search.query import Query

# Connect to Redis Stack (fix the port)
#redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=False)

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def sanitize_model_name(model_name):
    return model_name.replace("-", "_")

def search_embeddings(query, model_name, top_k=5, chunk_size_filter=None, overlap_filter=None):
    model_field = f"embedding_{sanitize_model_name(model_name)}"
    query_embedding = get_embedding(query, model_name)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Dynamically add filters
        filter_query = "*"
        if chunk_size_filter is not None:
            filter_query += f" @chunk_size:[{chunk_size_filter} {chunk_size_filter}]"
        if overlap_filter is not None:
            filter_query += f" @overlap:[{overlap_filter} {overlap_filter}]"

        q = (
            Query(f"{filter_query}=>[KNN {top_k} @{model_field} $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "chunk_size", "overlap", "vector_distance")
            .dialect(2)
        )

        results = redis_client.redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        top_results = [
            {
                "file": result.file.decode('utf-8'),
                "page": result.page,
                "chunk": result.chunk.decode('utf-8'),
                "chunk_size": result.chunk_size,
                "overlap": result.overlap,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ]

        for result in top_results:
            sim = float(result['similarity'])
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk Size: {result['chunk_size']}, Overlap: {result['overlap']}, Sim: {sim:.2f}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"File: {r.get('file')} | Page: {r.get('page')} | Chunk size: {r.get('chunk_size')} | Overlap: {r.get('overlap')} | Similarity: {float(r.get('similarity', 0)):.2f}\nText: {r.get('chunk')}"
            for r in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit\n")

    allowed_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "InstructorXL", "ollama-nomic"]
    model_name = input(
        f"Choose embedding model {allowed_models}: "
    )
    if model_name not in allowed_models:
        print(f"❌ Invalid model. Choose from {allowed_models}. Exiting.")
        return

    # Optional chunk filter
    chunk_size_filter = input("Optional: Specify chunk size to filter (or press Enter to skip): ")
    chunk_size_filter = int(chunk_size_filter) if chunk_size_filter.isdigit() else None

    overlap_filter = input("Optional: Specify overlap size to filter (or press Enter to skip): ")
    overlap_filter = int(overlap_filter) if overlap_filter.isdigit() else None

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, model_name, chunk_size_filter, overlap_filter)

        response = generate_rag_response(query, context_results)
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()