import redis
import numpy as np
import ollama
from redis.commands.search.query import Query
from embeddings import get_embedding as get_embedding_from_custom  # Import our custom embedding function

# Connect to Redis with decode_responses=True to automatically decode text
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

# Redis index configuration
INDEX_NAME = "embedding_index"
VECTOR_DIM = 768

# Specify the embedding model to use
EMBEDDING_MODEL = "InstructorXL"  # Use Instructor-XL model for better semantic search
print(f"🔍 Using embedding model: {EMBEDDING_MODEL}")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:
    """Get the embedding vector for text
    
    Prioritize using our custom embedding function, fall back to ollama if it fails
    """
    try:
        # Try using our custom embedding function
        embedding = get_embedding_from_custom(text, model)
        
        # If embedding is a nested list, take the first element
        # (Instructor-XL returns embeddings in a different format)
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
            
        print(f"✅ Generated embedding with {model} model, dimensions: {len(embedding)}")
        
    except Exception as e:
        print(f"⚠️ Custom embedding generation failed: {e}, falling back to ollama...")
        # Fall back to ollama
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
        print(f"✅ Generated embedding with ollama model, dimensions: {len(embedding)}")

    # Ensure correct dimensions
    if len(embedding) > VECTOR_DIM:
        print(f"⚠️ Truncating embedding: generated {len(embedding)} dimensions, Redis requires {VECTOR_DIM} dimensions")
        embedding = embedding[:VECTOR_DIM]
    elif len(embedding) < VECTOR_DIM:
        print(f"⚠️ Padding embedding: generated {len(embedding)} dimensions, Redis requires {VECTOR_DIM} dimensions")
        embedding += [0.0] * (VECTOR_DIM - len(embedding))

    return embedding

def search_embeddings(query, top_k=5):
    """Search for documents similar to the query using vector similarity"""
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct KNN search query
        q = (
            Query("*=>[KNN {} @embedding $vec AS vector_distance]".format(top_k))
            .sort_by("vector_distance")
            .return_fields("text", "vector_distance", "file", "page") 
            .dialect(2)
        )

        # Execute search
        results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})

        # Process results
        top_results = []
        for doc in results.docs:
            doc_dict = doc.__dict__  
            print(f"Redis returned keys: {doc_dict.keys()}")

            top_results.append({
                "file": doc_dict.get("file", "Unknown"),  
                "page": doc_dict.get("page", "Unknown"),  
                "chunk": doc_dict.get("text", ""),  
                "similarity": float(doc_dict.get("vector_distance", 0))  
            })

        # Display search results
        print("\n🔍 **Redis Search Results**:")
        for res in top_results:
            print(f"📄 File: {res['file']}, 📄 Page: {res['page']}\n📖 Content: {res['chunk'][:200]}...\n")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results):
    """Generate a response using RAG (Retrieval-Augmented Generation)"""
    if not context_results:
        return "I don't know. No relevant information found in the database."

    # Prepare context for the LLM
    context_str = "\n".join([
        f"From {res['file']} (Page {res['page']}):\n{res['chunk']}"
        for res in context_results
    ])

    print(f"📝 **Context passed to LLM:**\n{context_str[:500]}...\n")

    # Create prompt for the LLM
    prompt = f"""You are an AI assistant. Use the following context to answer the query.
    If the context is not relevant, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""
    
    # Generate response using Ollama
    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

def interactive_search():
    """Interactive search interface"""
    print("🔍 **RAG Search Interface**")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break
        
        # Search for relevant embeddings
        context_results = search_embeddings(query)
        
        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---\n", response)

if __name__ == "__main__":
    interactive_search()






 
 