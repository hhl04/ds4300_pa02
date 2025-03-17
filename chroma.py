import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import ollama
import numpy as np
import os
import fitz
import re

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

# Use a local SentenceTransformer as the embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # or another sentence-transformers model
)

collection_name = "foundations_pdf_collection"

# Create (or get if it already exists) the collection
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_fn
)

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

#TEXT PREP STRATEGY
def extract_clean_pdf(pdf_path):
    """Extract and clean text from a PDF file."""
    doc = fitz.open(pdf_path)
    cleaned_text = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        # Remove page numbers (assumes standalone numbers at the end of text)
        text = re.sub(r'\n\d+\n?$', '', text)
        
        # Replace special bullets and weird symbols
        text = text.replace("●", "-").replace("■", "-").replace("○", "-")
        
        # Remove unnecessary multiple newlines while keeping paragraph breaks
        text = re.sub(r'\n{2,}', '\n\n', text) 
        text = re.sub(r'\n+', ' ', text) 
        
        # Remove double spaces
        text = re.sub(r' +', ' ', text)

        # Fix encoding issues
        text = text.encode('utf-8', 'ignore').decode('utf-8')

        # Trim leading/trailing spaces
        cleaned_text.append(text.strip())  
    
    return cleaned_text

# split the text into chunks with overlap
# CHUNK SIZE and OVERLAP
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            
            #text_by_page = extract_text_from_pdf(pdf_path)
            text_by_page = extract_clean_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


# Add each chunk to Chroma
# Here, we generate a unique ID for each chunk; you can store metadata, too.
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        metadatas=[{"source": pdf_path, "chunk_index": i}],
        ids=[f"foundations-pdf-chunk-{i}"]
    )

print(f"Added {len(chunks)} chunks to collection '{collection_name}'.")
