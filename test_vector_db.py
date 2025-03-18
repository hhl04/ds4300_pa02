import os
import time
import numpy as np

# Import project components
from embeddings import get_embedding
from config import EMBEDDING_MODELS

# Import vector database interfaces
import redis
from redis.commands.search.query import Query
# If ChromaDB is installed
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not installed, related tests will be skipped")

# Set test data directory
DATA_DIR = "./ds4300 docs"
# Test questions
TEST_QUESTIONS = [
    "What is a NoSQL database?",
    "What are the main differences between Redis and MongoDB?",
    "How are relationships represented in Neo4j?",
]

def extract_clean_pdf(pdf_path):
    """Extract and clean text from a PDF file"""
    import fitz
    import re
    
    print(f"Processing PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        cleaned_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            # Remove page numbers (assuming they're standalone numbers at the end of text)
            text = re.sub(r'\n\d+\n?$', '', text)
            
            # Replace special symbols
            text = text.replace("●", "-").replace("■", "-").replace("○", "-")
            
            # Remove unnecessary multiple newlines while keeping paragraph breaks
            text = re.sub(r'\n{2,}', '\n\n', text) 
            text = re.sub(r'\n+', ' ', text) 
            
            # Remove double spaces
            text = re.sub(r' +', ' ', text)
    
            # Fix encoding issues
            text = text.encode('utf-8', 'ignore').decode('utf-8')
    
            # Trim leading/trailing spaces
            cleaned_text.append((page_num, text.strip()))  
        
        return cleaned_text
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        return []

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def prepare_test_data():
    """Prepare test data"""
    print("=== Preparing Test Data ===")
    
    # Prepare test data
    sample_pdf = os.path.join(DATA_DIR, "08 - PyMongo.pdf")
    if os.path.exists(sample_pdf):
        print(f"Extracting text from PDF file: {sample_pdf}")
        cleaned_text = extract_clean_pdf(sample_pdf)
        chunks = []
        for page_num, text in cleaned_text[:2]:  # Only use first 2 pages
            page_chunks = split_text_into_chunks(text, 500, 50)
            print(f"  Page {page_num}: Extracted {len(page_chunks)} text chunks")
            chunks.extend(page_chunks)
    else:
        # Create mock data
        print("PDF file not found, using mock data")
        chunks = [f"This is test chunk {i}, containing sample text for vector embedding testing. NoSQL databases are non-relational databases that don't require fixed table structures." for i in range(5)]
    
    print(f"Total prepared chunks: {len(chunks)}")
    
    # Generate embeddings
    model_name = "all-MiniLM-L6-v2"  # Default embedding model
    print(f"Generating embeddings for test data using model: {model_name}")
    
    embeddings = []
    for chunk in chunks:
        try:
            embedding = get_embedding(chunk, model_name)
            embeddings.append(embedding)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\nFailed to generate embedding: {str(e)}")
    print("")  # New line after progress dots
    
    print(f"Generated {len(embeddings)} embedding vectors")
    return chunks, embeddings

def test_redis(chunks, embeddings):
    """Test Redis vector database performance"""
    print("\n=== Testing Redis Vector Database ===")
    
    # Initialize Redis client
    try:
        redis_client = redis.Redis(host="localhost", port=6379, db=0)
        redis_client.ping()  # Test connection
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {str(e)}")
        return
    
    # Clear database
    redis_client.flushdb()
    print("Redis database cleared")
    
    # Create index
    try:
        VECTOR_DIM = len(embeddings[0]) if embeddings else 768
        INDEX_NAME = "embedding_index"
        DOC_PREFIX = "doc:"
        DISTANCE_METRIC = "COSINE"
        
        # Try to delete existing index
        try:
            redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
            print("Deleted existing index")
        except:
            pass
        
        # Create index
        start_time = time.time()
        redis_client.execute_command(
            f"""
            FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
            """
        )
        index_time = time.time() - start_time
        print(f"Redis index creation time: {index_time:.2f} seconds")
        
        # Store embeddings
        start_time = time.time()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            key = f"{DOC_PREFIX}{i}"
            redis_client.hset(
                key,
                mapping={
                    "text": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                }
            )
            if i % 2 == 0:  # Print progress every 2 chunks
                print(".", end="", flush=True)
        print("")  # New line after progress dots
        storage_time = time.time() - start_time
        print(f"Redis vector storage time: {storage_time:.2f} seconds")
        
        # Test queries
        print("\nExecuting test queries:")
        for i, question in enumerate(TEST_QUESTIONS):
            print(f"Query {i+1}: {question}")
            try:
                # Generate query embedding
                query_embedding = get_embedding(question, model_name="all-MiniLM-L6-v2")
                
                # Create query
                q = (
                    Query(f"*=>[KNN 3 @embedding $vec AS vector_distance]")
                    .sort_by("vector_distance")
                    .return_fields("text", "vector_distance")
                    .dialect(2)
                )
                
                # Execute query
                start_time = time.time()
                res = redis_client.ft(INDEX_NAME).search(
                    q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
                )
                query_time = time.time() - start_time
                
                print(f"  Query time: {query_time:.4f} seconds")
                print(f"  Found {len(res.docs)} results")
                
                # Print first result
                if res.docs:
                    distance = float(res.docs[0].vector_distance)
                    similarity = 1.0 - distance
                    print(f"  Best match (similarity {similarity:.4f}):")
                    print(f"  {res.docs[0].text[:100]}...")
            except Exception as e:
                print(f"  Query failed: {str(e)}")
        
        print("\nRedis vector database test completed")
    except Exception as e:
        print(f"Redis test failed: {str(e)}")

def test_chromadb(chunks, embeddings):
    """Test ChromaDB vector database performance"""
    print("\n=== Testing ChromaDB Vector Database ===")
    
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not installed, skipping test")
        return
    
    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        print("ChromaDB connection successful")
    except Exception as e:
        print(f"ChromaDB connection failed: {str(e)}")
        return
    
    # Create collection
    try:
        collection_name = "test_collection"
        
        # Delete existing collection (if exists)
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        start_time = time.time()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        index_time = time.time() - start_time
        print(f"ChromaDB collection creation time: {index_time:.2f} seconds")
        
        # Store embeddings
        start_time = time.time()
        ids = [f"id_{i}" for i in range(len(chunks))]
        metadatas = [{"source": f"document_{i//10}", "chunk": i} for i in range(len(chunks))]
        
        # Use batch add
        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings
        )
        storage_time = time.time() - start_time
        print(f"ChromaDB vector storage time: {storage_time:.2f} seconds")
        
        # Test queries
        print("\nExecuting test queries:")
        for i, question in enumerate(TEST_QUESTIONS):
            print(f"Query {i+1}: {question}")
            try:
                # Start timing
                start_time = time.time()
                results = collection.query(
                    query_texts=[question],
                    n_results=3
                )
                query_time = time.time() - start_time
                
                print(f"  Query time: {query_time:.4f} seconds")
                print(f"  Found {len(results['documents'][0])} results")
                
                # Print first result
                if results['documents'][0]:
                    print(f"  Best match:")
                    print(f"  {results['documents'][0][0][:100]}...")
            except Exception as e:
                print(f"  Query failed: {str(e)}")
        
        print("\nChromaDB vector database test completed")
    except Exception as e:
        print(f"ChromaDB test failed: {str(e)}")

def run_tests():
    """Run vector database tests without FAISS"""
    print("Starting simple vector database tests (Redis & ChromaDB)...")
    
    # Prepare test data
    chunks, embeddings = prepare_test_data()
    
    # Test Redis
    test_redis(chunks, embeddings)
    
    # Test ChromaDB
    test_chromadb(chunks, embeddings)
    
    print("\nVector database tests completed")

if __name__ == "__main__":
    run_tests() 