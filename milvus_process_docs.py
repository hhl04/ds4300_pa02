import numpy as np
import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from embeddings import get_embedding
from preprocessing import extract_clean_pdf, split_text_into_chunks
from config import EMBEDDING_MODELS
from tqdm import tqdm


# Initialize Milvus client
connections.connect("default", host="localhost", port="19530")

def create_collection(model_name):
    safe_model_name = model_name.replace("-", "_")
    collection_name = f"documents_{safe_model_name}"

    if utility.has_collection(collection_name):
        return Collection(name=collection_name)

    _, dim = EMBEDDING_MODELS[model_name]

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=50)
    ]

    schema = CollectionSchema(fields=fields, description=f"Document embeddings for {model_name}")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection

def clear_collections():
    for model_name in EMBEDDING_MODELS:
        collection_name = f"documents_{model_name.replace('-', '_')}"
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️ Dropped existing collection: {collection_name}")

def process_pdfs(data_dir):
    collections = {model_name: create_collection(model_name) for model_name in EMBEDDING_MODELS}

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    for model_name, (model, _) in EMBEDDING_MODELS.items():
        print(f"📥 Processing files for model: {model_name}")
        total_chunks = 0

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

                        collections[model_name].insert([
                            [doc_id],
                            [embedding],
                            [chunk],
                            [file_name],
                            [int(page_num)],
                            [model_name]
                        ])
                        pbar.update(1)

def create_vector_indexes():
    try:
        for model_name in EMBEDDING_MODELS:
            create_collection(model_name)
            print(f"✅ Milvus collection for {model_name} created or verified successfully!")
    except Exception as e:
        print(f"Error creating Milvus collections: {e}")

if __name__ == "__main__":
    clear_collections()
    create_vector_indexes()
    process_pdfs("ds4300 docs")
