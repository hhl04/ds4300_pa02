from redis_process_docs import redis_client
import numpy as np
from embeddings import get_embedding
import ollama
from redis.commands.search.query import Query

# Constants for Redis configuration
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Helper function to make model names safe for Redis index names
def sanitize_model_name(model_name):
    return model_name.replace("-", "_")

# Perform KNN vector search in Redis for a given query and embedding model
def search_embeddings(query, model_name, top_k=5):
    model_field = "embedding"
    
    # Get embedding for the search query
    query_embedding = get_embedding(query, model_name)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    
    # Build Redis index name based on model
    index_name = f"embedding_index_{sanitize_model_name(model_name)}"

    try:
        # Build Redis search query using vector similarity (KNN)
        q = (
            Query(f"*=>[KNN {top_k} @{model_field} $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "chunk_size", "overlap", "vector_distance")
            .dialect(2)
        )

        # Run search
        results = redis_client.ft(index_name).search(
            q, query_params={"vec": query_vector}
        )

        # Parse and clean up results
        top_results = [
            {
                "file": result.file.decode("utf-8") if isinstance(result.file, bytes) else result.file,
                "page": result.page,
                "chunk": result.chunk.decode("utf-8") if isinstance(result.chunk, bytes) else result.chunk,
                "chunk_size": result.chunk_size,
                "overlap": result.overlap,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ]

        # Print out results to user
        for result in top_results:
            sim = float(result["similarity"])
            print(
                f"---> File: {result['file']}, Page: {result['page']}, "
                f"Chunk Size: {result['chunk_size']}, Overlap: {result['overlap']}, Sim: {sim:.2f}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

# Generate a RAG (retrieval-augmented generation) response using Ollama LLM
def generate_rag_response(query, context_results):
    # Format the context from search results
    context_str = "\n".join(
        [
            f"File: {r.get('file')} | Page: {r.get('page')} | Chunk size: {r.get('chunk_size')} | Overlap: {r.get('overlap')} | Similarity: {float(r.get('similarity', 0)):.2f}\nText: {r.get('chunk')}"
            for r in context_results
        ]
    )

    # Construct prompt for the LLM
    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Get completion from the model
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Interactive CLI loop for querying the Redis vector store
def interactive_search():
    print("RAG Search Interface")
    print("Type 'exit' to quit\n")

    allowed_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "InstructorXL", "ollama-nomic"]
    model_name = input(f"Choose embedding model {allowed_models}: ")

    if model_name not in allowed_models:
        print(f"Invalid model. Choose from {allowed_models}. Exiting.")
        return

    # Prompt for queries and return RAG responses
    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, model_name)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)

# Entry point when script is run directly
if __name__ == "__main__":
    interactive_search()
