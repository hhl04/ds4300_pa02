import ollama
import redis
import numpy as np
import os
import fitz
import re
#from redis_client import store_embedding
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS
from embeddings import get_embedding

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
            # Remove punct
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
                    embedding = get_embedding(chunk)  
                    redis_key = f"doc:{file_name}:page{page_num}:chunk{chunk_index}"

                    redis_client.hset(redis_key, mapping={
                        "text": chunk,
                        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                        "file": file_name,  
                        "page": page_num    
                    })

            print(f"✅ Finished processing {file_name}")

def create_vector_index():
    """Check if the Redis vector index exists; if not, create it"""
    try:
        index_info = redis_client.execute_command("FT._LIST")
        if "embedding_index" not in index_info:
            redis_client.execute_command(
                "FT.CREATE embedding_index ON HASH PREFIX 1 doc: "
                "SCHEMA text TEXT "
                "file TEXT "
                "page NUMERIC "
                "embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE"
            )
            print("✅ Redis vector index created successfully!")
        else:
            print("✅ Redis vector index already exists, no need to create.")
    except Exception as e:
        print(f"❌ Error creating vector index: {e}")

if __name__ == "__main__":
    create_vector_index()
    process_pdfs("ds4300 docs")
