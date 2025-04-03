from pymilvus import connections, Collection
import numpy as np
import ollama
from config import EMBEDDING_MODELS

# Connect to Milvus instance
connections.connect("default", host="localhost", port="19530")

def get_embedding(text: str, model_name: str) -> list:
    """Generate an embedding for the input text using the specified model."""
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Model {model_name} is not recognized.")

    model, expected_dim = EMBEDDING_MODELS[model_name]

    if model_name == "ollama-nomic":
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response["embedding"]
    else:
        embedding = model.encode(text).tolist()

    if len(embedding) != expected_dim:
        raise ValueError(
            f"Error: Generated {len(embedding)} dimensions, expected {expected_dim} for {model_name}."
        )

    return embedding

def search_embeddings(query: str, model_name: str, top_k=3):
    """Search the Milvus collection for similar text chunks based on a query."""
    safe_model_name = model_name.replace("-", "_")
    collection_name = f"documents_{safe_model_name}"

    query_embedding = get_embedding(query, model_name)

    try:
        collection = Collection(name=collection_name)
        collection.load()

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "file", "page", "model"]
        )

        top_results = []
        if results:
            for hits in results:
                for hit in hits:
                    top_results.append({
                        "model": hit.get("model") or model_name,
                        "file": hit.get("file") or "Unknown",
                        "page": hit.get("page") or "Unknown",
                        "chunk": hit.get("text") or "",
                        "similarity": hit.score
                    })

        return top_results

    except Exception as e:
        print(f"Search error in {collection_name}: {e}")
        return []

def search_with_all_models(query, top_k=3):
    """Search across all models and return aggregated top-k results."""
    all_results = []
    for model_name in EMBEDDING_MODELS.keys():
        results = search_embeddings(query, model_name, top_k)
        all_results.extend(results)

    # Sort results by similarity score in descending order
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)

    print("\nAggregated Search Results from All Models:")
    for res in sorted_results[:top_k]:
        print(f"Model: {res['model']} | File: {res['file']}, Page: {res['page']}\n{res['chunk'][:200]}...\n")

    return sorted_results[:top_k]

def generate_rag_response(query, context_results):
    """Generate a response using retrieved context and a language model."""
    if not context_results:
        return "I don't know. No relevant information found in the database."

    context_str = "\n".join([
        f"From {res['file']} (Page {res['page']}, Model: {res['model']}):\n{res['chunk']}"
        for res in context_results
    ])

    prompt = f"""You are an AI assistant. Use the following context to answer the query.
If the context is not relevant, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
    """Run an interactive loop for querying the system."""
    print("RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_with_all_models(query)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---\n", response)

if __name__ == "__main__":
    interactive_search()
