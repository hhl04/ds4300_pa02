import chromadb
import numpy as np
import os
import fitz
import re
from config import EMBEDDING_MODELS, CHUNK_SIZES, OVERLAPS, VECTOR_INDEXES
from embeddings import get_embedding

# Initialize ChromaDB client with Docker connection parameters
# Default ChromaDB server typically runs on port 8000 in Docker
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

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
    """Iterate through PDFs in the directory, extract text, generate embeddings, and store them in ChromaDB"""
    # Create or get collections for each model
    collections = {}
    for model_name in EMBEDDING_MODELS:
        collections[model_name] = chroma_client.get_or_create_collection(
            name=f"documents_{model_name}",
            metadata={"hnsw:space": "cosine"}
        )

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            extracted_pages = extract_clean_pdf(pdf_path)

            for page_num, text in extracted_pages.items():
                chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)

                for chunk_index, chunk in enumerate(chunks):
                    for model_name, (model, _) in EMBEDDING_MODELS.items():
                        embedding = get_embedding(chunk, model_name=model_name, model=model)
                        
                        # Create unique ID for each chunk
                        doc_id = f"doc:{file_name}:page{page_num}:chunk{chunk_index}:{model_name}"
                        
                        # Add to ChromaDB collection
                        collections[model_name].add(
                            ids=[doc_id],
                            embeddings=[embedding],
                            metadatas=[{
                                "file": file_name,
                                "page": page_num,
                                "model": model_name
                            }],
                            documents=[chunk]
                        )

            print(f"✅ Finished processing {file_name}")

def create_vector_indexes():
    """Create separate ChromaDB collections for each embedding model"""
    try:
        for model_name in EMBEDDING_MODELS:
            collection_name = f"documents_{model_name}"
            # Collections are created automatically in process_pdfs
            # This just ensures they exist
            chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ ChromaDB collection {collection_name} created or verified successfully!")
    except Exception as e:
        print(f"Error creating ChromaDB collections: {e}")

if __name__ == "__main__":
    create_vector_indexes()
    process_pdfs("ds4300 docs")