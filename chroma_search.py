import chromadb
import numpy as np
import ollama
from config import EMBEDDING_MODELS

# Initialize ChromaDB client with Docker connection parameters
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Define vector dimensions for each model
VECTOR_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "InstructorXL": 768
}

def get_embedding(text: str, model_name: str) -> list:
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"❌ Model {model_name} is not recognized!")

    model, expected_dim = EMBEDDING_MODELS[model_name]
    embedding = model.encode(text).tolist()

    # Ensure correct embedding dimension
    if len(embedding) != expected_dim:
        raise ValueError(f"❌ Error: Generated {len(embedding)} dimensions, expected {expected_dim} for {model_name}!")

    return embedding


def search_embeddings(query: str, model_name: str, top_k=3):
    collection_name = f"documents_{model_name}"
    query_embedding = get_embedding(query, model_name)
    
    try:
        # Get the collection for this model
        collection = chroma_client.get_collection(name=collection_name)
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        top_results = []
        
        # Process the results
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                
                top_results.append({
                    "model": model_name,
                    "file": metadata.get("file", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "chunk": results["documents"][0][i],
                    "similarity": 1.0 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return top_results

    except Exception as e:
        print(f"❌ Search error in {collection_name}: {e}")
        return []

# Search using all models and aggregate results
def search_with_all_models(query, top_k=3):
    all_results = []
    for model in VECTOR_DIMS.keys():
        results = search_embeddings(query, model, top_k)
        all_results.extend(results)

    # Sort results by similarity score (higher is better)
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)

    print("\n🔍 **Aggregated Search Results from All Models**:")
    for res in sorted_results[:top_k]:  
        print(f"📄 Model: {res['model']} | File: {res['file']}, Page: {res['page']}\n📖 {res['chunk'][:200]}...\n")

    return sorted_results[:top_k]

# Generate a response using retrieved context
def generate_rag_response(query, context_results):
    if not context_results:
        return "I don't know. No relevant information found in the database."

    context_str = "\n".join([
        f"From {res['file']} (Page {res['page']}, Model: {res['model']}):\n{res['chunk']}"
        for res in context_results
    ])

    print(f"📝 **Context passed to LLM:**\n{context_str[:500]}...\n")

    prompt = f"""You are an AI assistant. Use the following context to answer the query.
    If the context is not relevant, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""

    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

# Interactive search interface
def interactive_search():
    print("🔍 **RAG Search Interface**")
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







