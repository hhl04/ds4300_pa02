# import ollama
# import redis
# import numpy as np
# import os
# import fitz
# import re
# #from redis_client import store_embedding
# from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS, VECTOR_INDEXES
# from embeddings import get_embedding

# redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

# def extract_clean_pdf(pdf_path, remove_pgnum=True, remove_sbullets=True, clean_formatting=True, remove_whitespace=True, remove_punct=True):
#     """Extract text from a PDF file and return a dictionary {page_number: cleaned_text}"""
#     doc = fitz.open(pdf_path)
#     extracted_text = {}

#     for page_num, page in enumerate(doc):
#         text = page.get_text("text")

#         if remove_pgnum:
#             # Remove page numbers (assumes standalone numbers at the end of text)
#             text = re.sub(r'\n\d+\n?$', '', text)

#         if remove_sbullets:
#             # Replace special bullets and weird symbols
#             text = text.replace("●", "-").replace("■", "-").replace("○", "-")

#         if clean_formatting:
#             # Remove unnecessary multiple newlines while keeping paragraph breaks
#             text = re.sub(r'\n{2,}', '\n\n', text)
#             text = re.sub(r'\n+', ' ', text)
            
#             # Remove double spaces
#             text = re.sub(r' +', ' ', text)
            
#             # Fix encoding issues
#             text = text.encode('utf-8', 'ignore').decode('utf-8')

#         if remove_punct:
#             # Remove punct
#             text = re.sub(r'[^\w\s]', '', text)

#         if remove_whitespace:
#             text = text.strip()

#         # Store only non-empty pages
#         if len(text) > 3:
#             extracted_text[page_num + 1] = text  

#     return extracted_text

# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     """Split text into chunks with overlap"""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks

# def process_pdfs(data_dir):
#     """Iterate through PDFs in the directory, extract text, generate embeddings, and store them in Redis"""
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(data_dir, file_name)
#             extracted_pages = extract_clean_pdf(pdf_path)

#             for page_num, text in extracted_pages.items():
#                 chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)

#                 for chunk_index, chunk in enumerate(chunks):
#                     for model_name, (model, dim) in EMBEDDING_MODELS.items():
#                         embedding = get_embedding(chunk, model_name)  
#                         redis_key = f"doc:{file_name}:page{page_num}:chunk{chunk_index}:{model_name}"

#                         redis_client.hset(redis_key, mapping={
#                             "text": chunk,
#                             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#                             "file": file_name,  
#                             "page": page_num,
#                             "model": model_name  
#                         })

#             print(f"✅ Finished processing {file_name}")

# def create_vector_indexes():
#     """Create separate Redis vector indexes for each embedding model"""
#     try:
#         index_info = redis_client.execute_command("FT._LIST")

#         for model_name, index_name in VECTOR_INDEXES.items():
#             _, dim = EMBEDDING_MODELS[model_name]  # Get correct dimension
#             if index_name not in index_info:
#                 redis_client.execute_command(
#                     f"FT.CREATE {index_name} ON HASH PREFIX 1 doc: "
#                     f"SCHEMA text TEXT "
#                     f"file TEXT "
#                     f"page NUMERIC "
#                     f"model TEXT "
#                     f"embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {dim} DISTANCE_METRIC COSINE"
#                 )
#                 print(f"✅ Redis vector index {index_name} created successfully!")
#             else:
#                 print(f"✅ Redis vector index {index_name} already exists, no need to create.")
#     except Exception as e:
#         print(f"Error creating vector indexes: {e}")

# if __name__ == "__main__":
#     create_vector_indexes()
#     process_pdfs("ds4300 docs")

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


# # split the text into chunks with overlap
# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     """Split text into chunks of approximately chunk_size words with overlap."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks

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

# # Create multiple text chuncks, with varying sizes and overlaps
# def split_text_variants(text, chunk_sizes=[200, 500, 1000], overlaps=[0, 50, 100]):
#     split_variants = []
#     for chunk_size in chunk_sizes:
#         for overlap in overlaps:
#             chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
#             split_variants.append((chunk_size, overlap, chunks))
    # return split_variants

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

# def process_pdfs(data_dir):
#     all_chunks = []

#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(data_dir, file_name)
#             cleaned_pages = extract_clean_pdf(pdf_path)

#             for page_num, text in enumerate(cleaned_pages):
#                 split_variants = split_text_variants(
#                     text, 
#                     chunk_sizes=CHUNK_SIZES, 
#                     overlaps=OVERLAPS
#                 )

#                 for chunk_size, overlap, chunks in split_variants:
#                     for chunk_index, chunk in enumerate(chunks):
#                         all_chunks.append({
#                             "file": file_name,
#                             "page": page_num,
#                             "chunk": chunk,
#                             "chunk_index": chunk_index,
#                             "chunk_size": chunk_size,
#                             "overlap": overlap
#                         })
#             print(f"Finished processing {file_name}")

#     return all_chunks

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

    # # Test split_text_into_chunks
    # dummy_text = "This is a simple test sentence to verify that text chunking works as expected. " * 10

    # print("\n Testing split_text_into_chunks()")
    # chunks = split_text_into_chunks(dummy_text, chunk_size=10, overlap=2)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i}: {chunk}")

    # # Test split_text_variants
    # print("\nTesting split_text_variants()...")
    # variants = split_text_variants(dummy_text, chunk_sizes=[10, 20], overlaps=[0, 5])
    # for chunk_size, overlap, chunks in variants:
    #     print(f"\n--- Variant (size={chunk_size}, overlap={overlap}) ---")
    #     for i, chunk in enumerate(chunks):
    #         print(f"Chunk {i}: {chunk}")

    # Test process_pdfs()
    # print('\nTesting process_pdfs()')
    # test_dir = "./ds4300 docs"
    # process_pdfs(test_dir)
