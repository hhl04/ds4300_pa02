import time
import psutil
import os
import chromadb
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging
import json
from PyPDF2 import PdfReader
import redis
from redis.commands.search.query import Query
from pymilvus import connections, Collection
import ollama
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveComparison:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        # LLM models
        self.llm_models = {
            "Mistral": "mistral",
            "Llama": "llama2"
        }
        # Embedding models
        self.embedding_models = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
            "InstructorXL": "hkunlp/instructor-xl"
        }
        self.chunk_sizes = [200, 500, 1000]
        self.chunk_overlaps = [0, 50, 100]
        self.vector_dbs = ["ChromaDB", "Redis", "Milvus"]
        self.results = {}
        
        # Fixed test questions
        self.test_queries = [
            "What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?",
            "When are linked lists faster than contiguously-allocated lists?",
            "Why is a B+ Tree a better than an AVL tree when indexing a large dataset?",
            "What is disk-based indexing and why is it important for database systems?",
            "In the context of a relational database system, what is a transaction?",
            "Succinctly describe the four components of ACID compliant transactions.",
            "Why does the CAP principle not make sense when applied to a single-node MongoDB instance?",
            "Describe the differences between horizontal and vertical scaling.",
            "Briefly describe how a key/value store can be used as a feature store.",
            "When was Redis originally released?",
            "In Redis, what is the difference between the INC and INCR commands?",
            "What are the benefits of BSON over JSON in MongoDB?",
            "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre?",
            "What does the $nin operator mean in a Mongo query?"
        ]
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        
        # Initialize Milvus connection
        connections.connect("default", host="localhost", port="19530")
        
        # Initialize embedding models
        self.sentence_transformers = {
            name: SentenceTransformer(model_path)
            for name, model_path in self.embedding_models.items()
        }
        
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
        
    def load_pdf_documents(self):
        documents = []
        pdf_dir = "./ds4300 docs"
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(text)
        return documents
        
    def create_chunks(self, text, chunk_size, overlap):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
        
    def get_embedding(self, text, model_name):
        """Get text embedding"""
        if model_name in self.llm_models:
            # Use Ollama LLM
            response = ollama.embeddings(model=self.llm_models[model_name], prompt=text)
            return response['embedding']
        else:
            # Use SentenceTransformer
            model = self.sentence_transformers[model_name]
            return model.encode(text)
        
    def test_vector_db(self, db_name, model_name, chunk_size, overlap):
        collection_name = f"test_{chunk_size}_{overlap}_{model_name}_{db_name}"
        
        # Load documents and create chunks
        documents = self.load_pdf_documents()
        chunks = []
        for doc in documents:
            chunks.extend(self.create_chunks(doc, chunk_size, overlap))
        
        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk, model_name)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        if db_name == "ChromaDB":
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.create_collection(name=collection_name)
            
            # Index documents
            start_time = time.time()
            collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=[f"doc_{i}" for i in range(len(chunks))]
            )
            indexing_time = time.time() - start_time
            
        elif db_name == "Redis":
            # Create Redis index
            start_time = time.time()
            VECTOR_DIM = len(embeddings[0])
            INDEX_NAME = collection_name
            DOC_PREFIX = "doc:"
            
            # Create index
            self.redis_client.execute_command(
                f"""
                FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
                SCHEMA text TEXT
                embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC COSINE
                """
            )
            
            # Store embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                key = f"{DOC_PREFIX}{i}"
                self.redis_client.hset(
                    key,
                    mapping={
                        "text": chunk,
                        "embedding": embeddings[i].tobytes(),
                    }
                )
            indexing_time = time.time() - start_time
            
        elif db_name == "Milvus":
            # Create Milvus collection
            start_time = time.time()
            collection = Collection(
                name=collection_name,
                schema={
                    "fields": [
                        {"name": "id", "dtype": "INT64", "is_primary": True},
                        {"name": "text", "dtype": "VARCHAR", "max_length": 65535},
                        {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": len(embeddings[0])}
                    ]
                }
            )
            
            # Insert data
            entities = [
                [i for i in range(len(chunks))],  # ids
                chunks,  # text
                embeddings.tolist()  # embeddings
            ]
            collection.insert(entities)
            collection.flush()
            indexing_time = time.time() - start_time
        
        # Test queries
        query_results = []
        for query in self.test_queries:
            query_embedding = self.get_embedding(query, model_name)
            start_time = time.time()
            
            if db_name == "ChromaDB":
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3
                )
                query_results.append({
                    "query": query,
                    "results": {
                        "documents": results["documents"][0],
                        "distances": results["distances"][0],
                        "query_time": time.time() - start_time
                    }
                })
                
            elif db_name == "Redis":
                q = (
                    Query(f"*=>[KNN 3 @embedding $vec AS vector_distance]")
                    .sort_by("vector_distance")
                    .return_fields("text", "vector_distance")
                    .dialect(2)
                )
                res = self.redis_client.ft(INDEX_NAME).search(
                    q, query_params={"vec": query_embedding.tobytes()}
                )
                query_results.append({
                    "query": query,
                    "results": {
                        "documents": [doc.text for doc in res.docs],
                        "distances": [float(doc.vector_distance) for doc in res.docs],
                        "query_time": time.time() - start_time
                    }
                })
                
            elif db_name == "Milvus":
                collection.load()
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                }
                results = collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=3,
                    output_fields=["text"]
                )
                query_results.append({
                    "query": query,
                    "results": {
                        "documents": [hit.entity.get('text') for hit in results[0]],
                        "distances": [hit.distance for hit in results[0]],
                        "query_time": time.time() - start_time
                    }
                })
        
        return {
            "indexing": {
                "time": indexing_time,
                "memory_usage": self.get_memory_usage(),
                "document_count": len(chunks)
            },
            "querying": query_results
        }
        
    def evaluate_combination(self, results):
        """Evaluate the performance of a combination"""
        if not results.get("success"):
            return float("-inf")
            
        indexing_time = results["indexing"]["time"]
        avg_query_time = sum(q["results"]["query_time"] for q in results["querying"]) / len(results["querying"])
        memory_usage = results["indexing"]["memory_usage"]
        
        # Calculate comprehensive score (lower is better)
        return - (indexing_time + avg_query_time + memory_usage)
        
    def find_best_chunk_size(self, model_name, db_name):
        """Find the best chunk size"""
        logger.info(f"Finding best chunk size for {model_name} on {db_name}")
        best_score = float("-inf")
        best_chunk_size = None
        
        for chunk_size in self.chunk_sizes:
            # Use default overlap
            overlap = 50
            key = f"{chunk_size}_{overlap}_{model_name}_{db_name}"
            
            try:
                results = self.test_vector_db(db_name, model_name, chunk_size, overlap)
                self.results[key] = {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_name": model_name,
                    "model_type": "LLM" if model_name in self.llm_models else "Embedding",
                    "db_name": db_name,
                    **results,
                    "success": True
                }
                
                score = self.evaluate_combination(self.results[key])
                if score > best_score:
                    best_score = score
                    best_chunk_size = chunk_size
                    
            except Exception as e:
                logger.error(f"Error testing chunk size {chunk_size}: {str(e)}")
                self.results[key] = {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_name": model_name,
                    "model_type": "LLM" if model_name in self.llm_models else "Embedding",
                    "db_name": db_name,
                    "error": str(e),
                    "success": False
                }
                
        return best_chunk_size
        
    def find_best_overlap(self, model_name, db_name, chunk_size):
        """Find the best overlap size"""
        logger.info(f"Finding best overlap for {model_name} on {db_name} with chunk size {chunk_size}")
        best_score = float("-inf")
        best_overlap = None
        
        for overlap in self.chunk_overlaps:
            key = f"{chunk_size}_{overlap}_{model_name}_{db_name}"
            
            try:
                results = self.test_vector_db(db_name, model_name, chunk_size, overlap)
                self.results[key] = {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_name": model_name,
                    "model_type": "LLM" if model_name in self.llm_models else "Embedding",
                    "db_name": db_name,
                    **results,
                    "success": True
                }
                
                score = self.evaluate_combination(self.results[key])
                if score > best_score:
                    best_score = score
                    best_overlap = overlap
                    
            except Exception as e:
                logger.error(f"Error testing overlap {overlap}: {str(e)}")
                self.results[key] = {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_name": model_name,
                    "model_type": "LLM" if model_name in self.llm_models else "Embedding",
                    "db_name": db_name,
                    "error": str(e),
                    "success": False
                }
                
        return best_overlap
        
    def run_comparison(self):
        # Find best chunk size for each model and database combination
        best_combinations = {}
        
        # Test LLM models
        for model_name in self.llm_models:
            for db_name in self.vector_dbs:
                best_chunk_size = self.find_best_chunk_size(model_name, db_name)
                if best_chunk_size:
                    best_overlap = self.find_best_overlap(model_name, db_name, best_chunk_size)
                    best_combinations[f"{model_name}_{db_name}"] = {
                        "chunk_size": best_chunk_size,
                        "overlap": best_overlap
                    }
        
        # Test embedding models
        for model_name in self.embedding_models:
            for db_name in self.vector_dbs:
                best_chunk_size = self.find_best_chunk_size(model_name, db_name)
                if best_chunk_size:
                    best_overlap = self.find_best_overlap(model_name, db_name, best_chunk_size)
                    best_combinations[f"{model_name}_{db_name}"] = {
                        "chunk_size": best_chunk_size,
                        "overlap": best_overlap
                    }
        
        # Save results
        with open("comprehensive_comparison_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        # Save best combinations
        self.results["best_combinations"] = best_combinations
        with open("comprehensive_comparison_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    comparison = ComprehensiveComparison()
    comparison.run_comparison()