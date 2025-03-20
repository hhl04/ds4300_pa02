import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import ollama
import numpy as np
import os
#import fitz
import re

# Import specific functions from process_docs.py
from process_docs import process_pdfs, extract_clean_pdf, split_text_into_chunks, split_text_variants, preprocess_text
from embeddings import get_embedding, benchmark_embedding  # Import from embeddings.py

class MyChromaEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name

    def __call__(self, texts):
        """
        Chroma will pass in a list of strings (texts),
        and expects a list of embeddings (list of float lists) back.
        """
        return [get_embedding(t, self.model_name) for t in texts]


# Initialize Chroma connection
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_client.heartbeat()

# Initialize Chroma client with local persistence
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"  # folder to persist DB
    )
)

# use the get_embedding function from embeddings.py
my_embedding_fn = MyChromaEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create (or get if it already exists) the collection
collection = client.get_or_create_collection(
    name="midterm_study_guide",
    embedding_function=my_embedding_fn
)

def add_documents_to_chroma(chunks):
    """
    Add processed document chunks to ChromaDB collection.
    
    Args:
        chunks: List of chunk dictionaries from process_pdfs()
    """
    if not chunks:
        print("No chunks to add to the collection")
        return
    
    # Prepare the data for ChromaDB format
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}_{chunk['file']}_{chunk['page']}_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk['chunk'])
        metadatas.append({
            "file": chunk['file'],
            "page": chunk['page'],
            "chunk_size": chunk['chunk_size'],
            "overlap": chunk['overlap'],
            "chunk_index": chunk['chunk_index']
        })
    
    # Add data to collection in batches (to avoid potential size limitations)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    print(f"Added {len(ids)} chunks to ChromaDB collection")

def query_collection(query_text, n_results=3):
    """
    Query the ChromaDB collection with a string question.
    
    Args:
        query_text: The question or query text
        n_results: Number of results to return (default: 3)
    
    Returns:
        The query results from ChromaDB
    """
    processed_query = preprocess_text(query_text, strategy='standard')
    
    query_results = collection.query(
        query_texts=[processed_query],
        n_results=n_results
    )
    
    return query_results

def main():
    # Process PDFs and get chunks
    pdf_dir = "/Users/huytuonghoangle/Documents/GitHub/ds4300_pa02/ds4300 docs"
    print(f"Processing PDFs from {pdf_dir}...")
    chunks = process_pdfs(pdf_dir)
    
    # Add chunks to ChromaDB
    print("Adding chunks to ChromaDB...")
    add_documents_to_chroma(chunks)
    
    # Example query
    example_query = "What are the benefits of using MongoDB?"
    print(f"\nQuerying: '{example_query}'")
    results = query_collection(example_query, n_results=3)
    
    # Display results nicely
    print("\nResults:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {metadata['file']}, Page: {metadata['page']}")
        print(f"Chunk: {doc[:200]}...")  # Show first 200 chars
    
    # Interactive query mode
    while True:
        user_query = input("\nEnter a query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        results = query_collection(user_query)
        
        # Display results
        print("\nResults:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {metadata['file']}, Page: {metadata['page']}")
            print(f"Chunk: {doc[:200]}...")  # Show first 200 chars

if __name__ == "__main__":
    main()