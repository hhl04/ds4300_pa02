import chromadb
import numpy as np
import ollama
from config import EMBEDDING_MODELS

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def get_embedding(text: str, model_name: str) -> list:
    """
    Generate an embedding for a given text using the specified model.
    """
    model_config = EMBEDDING_MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Model {model_name} is not recognized.")

    model, expected_dim = model_config

    if model_name == "ollama-nomic":
        # Use Ollama to generate embedding
        response = ollama.embeddings(model=model, prompt=text)
        embedding = response["embedding"]
    else:
        # Use SentenceTransformer
        embedding = model.encode(text).tolist()

    if len(embedding) != expected_dim:
        raise ValueError(f"Generated {len(embedding)} dimensions, expected {expected_dim} for {model_name}.")

    return embedding

def search_embeddings(query: str, model_name: str, top_k=3):
    """
    Search a ChromaDB collection using the embedding of the query.
    """
    collection_name = f"documents_{model_name}"
    query_embedding = get_embedding(query, model_name)

    try:
        collection = chroma_client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        top_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                top_results.append({
                    "model": model_name,
                    "file": metadata.get("file", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "chunk": results["documents"][0][i],
                    "similarity": 1.0 - results["distances"][0][i]
                })

        return top_results

    except Exception as e:
        print(f"Search error in {collection_name}: {e}")
        return []

def search_with_all_models(query, top_k=3):
    """
    Search across all models and return the most relevant results.
    """
    all_results = []

    for model_name in EMBEDDING_MODELS.keys():
        results = search_embeddings(query, model_name, top_k)
        for res in results:
            res["model"] = model_name
        all_results.extend(results)

    # Sort by similarity score in descending order
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
    return sorted_results[:top_k]

def generate_rag_response(query, context_results):
    """
    Use top retrieved chunks to generate an answer using a language model.
    """
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

    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search():
    """
    Simple interactive loop for entering queries and viewing results.
    """
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
