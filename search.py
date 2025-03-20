import redis
import numpy as np
import ollama
from redis.commands.search.query import Query
from config import EMBEDDING_MODELS, VECTOR_INDEXES

redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

INDEX_384 = "embedding_index_384"  # For MiniLM (384 dimensions)
INDEX_768 = "embedding_index_768"  # For mpnet-base & InstructorXL (768 dimensions)

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
    index_name = VECTOR_INDEXES[model_name]
    query_embedding = get_embedding(query, model_name)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN {} @embedding $vec AS vector_distance]".format(top_k))
            .sort_by("vector_distance")
            .return_fields("text", "vector_distance", "file", "page")
            .dialect(2)
        )

        results = redis_client.ft(index_name).search(q, query_params={"vec": query_vector})

        top_results = []
        for doc in results.docs:
            doc_dict = doc.__dict__
            print(f"🔍 Redis returned keys: {doc_dict.keys()}")

            top_results.append({
                "model": model_name,
                "file": doc_dict.get("file", "Unknown"),
                "page": doc_dict.get("page", "Unknown"),
                "chunk": doc_dict.get("text", ""),
                "similarity": float(doc_dict.get("vector_distance", 0))
            })

        return top_results

    except Exception as e:
        print(f"❌ Search error in {index_name}: {e}")
        return []

# Search using all models and aggregate results
def search_with_all_models(query, top_k=3):
    all_results = []
    for model in VECTOR_DIMS.keys():
        results = search_embeddings(query, model, top_k)
        all_results.extend(results)

    # Sort results by similarity score
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=False)

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






 
 