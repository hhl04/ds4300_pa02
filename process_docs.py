import ollama
import redis
import numpy as np
import os
import fitz
import re
#from redis_client import store_embedding
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS, VECTOR_INDEXES
from embeddings import get_embedding

import time
import nltk
from nltk.corpus import stopwords
import sys

redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)


def extract_clean_pdf(pdf_path, remove_pgnum=True, remove_sbullets=True, clean_formatting=True, remove_whitespace=True, remove_punct=True):
    """Extract text from a PDF file and return a dictionary {page_number: cleaned_text}"""

    doc = fitz.open(pdf_path)
    extracted_text = {}

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
            # Only remove punctuation if explicitly requested
            # Be careful with code - preferably don't use this option for code-heavy documents
            if preserve_code:
                # Identify potential code blocks to preserve
                lines = text.split('\n')
                processed_lines = []
                
                for line in lines:
                    code_indicators = ['import ', 'from ', ' = ', '(', ')', 'def ', 'class ', '{}', '[]', '+=', '-=', '*=', '/=']
                    is_likely_code = any(indicator in line for indicator in code_indicators)
                    
                    if is_likely_code:
                        processed_lines.append(line)  # Preserve code lines as-is
                    else:
                        # Remove punctuation from non-code lines
                        processed_line = re.sub(r'[^\w\s]', '', line)
                        processed_lines.append(processed_line)
                
                text = '\n'.join(processed_lines)
            else:
                # Remove all punctuation
                text = re.sub(r'[^\w\s]', '', text)

        if remove_whitespace:
            text = text.strip()

        # Store only non-empty pages
        if len(text) > 3:
            extracted_text[page_num + 1] = text  


    return extracted_text


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir):
    """Iterate through PDFs in the directory, extract text, generate embeddings, and store them in Redis"""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            extracted_pages = extract_clean_pdf(pdf_path)

            for page_num, text in extracted_pages.items():
                chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)

                for chunk_index, chunk in enumerate(chunks):
                    for model_name, (model, dim) in EMBEDDING_MODELS.items():
                        embedding = get_embedding(chunk, model_name)  
                        redis_key = f"doc:{file_name}:page{page_num}:chunk{chunk_index}:{model_name}"

                        redis_client.hset(redis_key, mapping={
                            "text": chunk,
                            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                            "file": file_name,  
                            "page": page_num,
                            "model": model_name  
                        })

            print(f"✅ Finished processing {file_name}")


def create_vector_indexes():
    """Create separate Redis vector indexes for each embedding model"""
    try:
        index_info = redis_client.execute_command("FT._LIST")

        for model_name, index_name in VECTOR_INDEXES.items():
            _, dim = EMBEDDING_MODELS[model_name]  # Get correct dimension
            if index_name not in index_info:
                redis_client.execute_command(
                    f"FT.CREATE {index_name} ON HASH PREFIX 1 doc: "
                    f"SCHEMA text TEXT "
                    f"file TEXT "
                    f"page NUMERIC "
                    f"model TEXT "
                    f"embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {dim} DISTANCE_METRIC COSINE"
                )
                print(f"✅ Redis vector index {index_name} created successfully!")
            else:
                print(f"✅ Redis vector index {index_name} already exists, no need to create.")
    except Exception as e:
        print(f"Error creating vector indexes: {e}")

if __name__ == "__main__":
    create_vector_indexes()
    process_pdfs("ds4300 docs")
