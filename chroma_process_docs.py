import chromadb
import numpy as np
import os
from config import EMBEDDING_MODELS
from embeddings import get_embedding
from preprocessing import extract_clean_pdf, split_text_into_chunks
from tqdm import tqdm

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def clear_collections():
    """Delete existing ChromaDB collections for all embedding models"""
    for model_name in EMBEDDING_MODELS:
        collection_name = f"documents_{model_name}"
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"🗑️ Cleared collection: {collection_name}")
        except Exception as e:
            print(f"⚠️ Could not clear collection {collection_name}: {e}")

def process_pdfs(data_dir):
    """Iterate through PDFs, extract text, generate embeddings, and store them in ChromaDB"""
    # Create or get collections for each model
    collections = {
        model_name: chroma_client.get_or_create_collection(
            name=f"documents_{model_name}",
            metadata={"hnsw:space": "cosine"}
        )
        for model_name in EMBEDDING_MODELS
    }

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    for model_name, (model, _) in EMBEDDING_MODELS.items():
        print(f"📥 Processing files for model: {model_name}")
        total_chunks = 0

        # First, count total chunks for progress bar
        for file_name in pdf_files:
            pdf_path = os.path.join(data_dir, file_name)
            extracted_pages = extract_clean_pdf(pdf_path)
            for text in extracted_pages.values():
                total_chunks += len(split_text_into_chunks(text))

        with tqdm(total=total_chunks, desc=f"Storing chunks for {model_name}") as pbar:
            for file_name in pdf_files:
                pdf_path = os.path.join(data_dir, file_name)
                extracted_pages = extract_clean_pdf(pdf_path)

                for page_num, text in extracted_pages.items():
                    chunks = split_text_into_chunks(text)

                    for chunk_index, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk, model_name=model_name, model=model)
                        doc_id = f"doc:{file_name}:page{page_num}:chunk{chunk_index}:{model_name}"

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
                        pbar.update(1)

def create_vector_indexes():
    """Ensure ChromaDB collections exist for each embedding model"""
    try:
        for model_name in EMBEDDING_MODELS:
            chroma_client.get_or_create_collection(
                name=f"documents_{model_name}",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ ChromaDB collection documents_{model_name} created or verified successfully!")
    except Exception as e:
        print(f"Error creating ChromaDB collections: {e}")

if __name__ == "__main__":
    clear_collections()
    create_vector_indexes()
    process_pdfs("ds4300 docs")
