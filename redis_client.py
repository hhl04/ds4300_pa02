import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
from process_docs import process_pdfs

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")

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


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


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


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main():
    clear_redis_store()
    create_hnsw_index()

    #REPLACE WITH YOUR DIRECTORY
    process_pdfs("/Users/huytuonghoangle/Documents/GitHub/ds4300_pa02/ds4300 docs")

    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()
