import ollama
import numpy as np
import os
import fitz
import re
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS, VECTOR_INDEXES
from embeddings import get_embedding

# Initialize Milvus client with Docker connection parameters
# Default Milvus server typically runs on port 19530 in Docker
connections.connect("default", host="localhost", port="19530")

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

def create_collection(model_name):
    """Create a Milvus collection for a specific embedding model"""
    collection_name = f"documents_{model_name}"
    
    # Skip if collection already exists
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
    
    # Get dimension for this model
    _, dim = EMBEDDING_MODELS[model_name]
    
    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=50)
    ]
    
    # Create schema and collection
    schema = CollectionSchema(fields=fields, description=f"Document embeddings for {model_name}")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create an IVF_FLAT index for vector field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

def process_pdfs(data_dir):
    """Iterate through PDFs in the directory, extract text, generate embeddings, and store them in Milvus"""
    # Create collections for each model
    collections = {}
    for model_name in EMBEDDING_MODELS:
        collections[model_name] = create_collection(model_name)

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            extracted_pages = extract_clean_pdf(pdf_path)

            for page_num, text in extracted_pages.items():
                chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)

                for chunk_index, chunk in enumerate(chunks):
                    for model_name in EMBEDDING_MODELS:
                        embedding = get_embedding(chunk, model_name)
                        
                        # Create unique ID for each chunk
                        doc_id = f"doc:{file_name}:page{page_num}:chunk{chunk_index}:{model_name}"
                        
                        # Add to Milvus collection
                        collection = collections[model_name]
                        collection.insert([
                            [doc_id],              # id
                            [embedding],           # embedding
                            [chunk],               # text
                            [file_name],           # file
                            [int(page_num)],       # page
                            [model_name]           # model
                        ])

            print(f"✅ Finished processing {file_name}")

def create_vector_indexes():
    """Create separate Milvus collections for each embedding model"""
    try:
        for model_name in EMBEDDING_MODELS:
            collection = create_collection(model_name)
            print(f"✅ Milvus collection for {model_name} created or verified successfully!")
    except Exception as e:
        print(f"Error creating Milvus collections: {e}")

if __name__ == "__main__":
    create_vector_indexes()
    process_pdfs("ds4300 docs")