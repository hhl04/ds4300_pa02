import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
#from redis_client import store_embedding
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS
from embeddings import get_embedding, benchmark_embedding

redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

def extract_clean_pdf(pdf_path,remove_pgnum=True, remove_sbullets=True, clean_formatting=True, remove_whitespace=True, remove_punct=False):
    """Extract and clean text from a PDF file."""
    doc = fitz.open(pdf_path)
    cleaned_text = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        if remove_pgnum:
            # Remove page numbers (assumes standalone numbers at the end of text)
                text = re.sub(r'\n\d+\n?$', '', text)
        
        if remove_sbullets:
            # Replace special bullets and weird symbols
            text = text.replace("●", "-").replace("■", "-").replace("○", "-")
        
        if clean_formatting:
            # Remove unnecessary multiple newlines while keeping paragraph breaks
            text = re.sub(r'\n{2,}', '\n\n', text) 
            text = re.sub(r'\n+', ' ', text) 
        
            # Remove double spaces
            text = re.sub(r' +', ' ', text)

            # Fix encoding issues
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        if remove_punct:
            # Remove punct
            text = re.sub(r'[^\w\s]', '', text)

        if remove_whitespace:
            text = text.strip()

        if len(text) > 3:
            cleaned_text.append(text)
    
    return cleaned_text

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append({
            "chunk": chunk,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "start_idx": i,  # Optional: can help debug or inspect token ranges
            "end_idx": min(i + chunk_size, len(words))
        })
    return chunks

def split_text_variants(text, file_name, page_num, chunk_sizes=[200, 500, 1000], overlaps=[0, 50, 100]):
    split_variants = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
            for chunk_index, chunk in enumerate(chunks):
                split_variants.append({
                    "file": file_name,
                    "page": page_num,
                    "chunk": chunk,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                    "overlap": overlap
                })
    return split_variants

def process_pdfs(data_dir):
    all_chunks = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            cleaned_pages = extract_clean_pdf(pdf_path)

            for page_num, text in enumerate(cleaned_pages):
                split_variants = split_text_variants(
                    text, 
                    file_name=file_name,
                    page_num=page_num,
                    chunk_sizes=CHUNK_SIZES, 
                    overlaps=OVERLAPS
                )
                all_chunks.extend(split_variants)

            print(f"Finished processing {file_name}")

    return all_chunks


def create_vector_index():
    """
    Checks if the Redis vector index exists; if not, creates it.
    """
    try:
        # Check if the index exists
        index_info = redis_client.execute_command("FT._LIST")
        if "embedding_index" not in index_info:
            redis_client.execute_command(
                "FT.CREATE embedding_index ON HASH PREFIX 1 doc: "
                "SCHEMA text TEXT "
                "embedding VECTOR HNSW 6 DIM 768 TYPE FLOAT32 DISTANCE_METRIC COSINE"
            )
            print("Redis vector index created successfully!")
        else:
            print("Redis vector index already exists.")
    except Exception as e:
        print(f"Error creating vector index: {e}")


# TESTING 
if __name__ == "__main__":
    
    # create_vector_index()

    print("\n Testing extract_clean_pdf()")
    test_pdf = "./ds4300 docs/Document_DBs_&_MongoDB_Study_Guide.pdf"
    clean_pdf = extract_clean_pdf(test_pdf, remove_punct=False)
    print(clean_pdf)