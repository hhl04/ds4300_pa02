import redis_client  # assuming this has the redis schema setup etc.
import json
import numpy as np
from embeddings import get_embedding
import ollama
from redis.commands.search.query import Query
import redis

# Connect to Redis Stack (fix the port)
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=False)

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def sanitize_model_name(model_name):
    return model_name.replace("-", "_")

def search_embeddings(query, model_name, top_k=5):
    model_field = f"embedding_{sanitize_model_name(model_name)}"
    query_embedding = get_embedding(query, model_name)

    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query(f"*=>[KNN {top_k} @{model_field} $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ]

        for result in top_results:
            sim = float(result['similarity'])
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Sim: {sim:.2f}")


        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    # Injecting context into prompt
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
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    # Ask for embedding model once at the start
    model_name = input(
        "Choose embedding model (all-MiniLM-L6-v2, all-mpnet-base-v2, InstructorXL, ollama-nomic): "
    )

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, model_name)

        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()