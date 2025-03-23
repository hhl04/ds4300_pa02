# DS4300 Programming Assignment 02 - Vector Database with Redis Stack

This project implements a vector database using Redis Stack and RediSearch for semantic search over PDF documents. The system includes document processing, vector embedding generation, and a search interface.

## System Requirements

- Python 3.8+
- Redis Stack (via Docker)
- Required Python packages (see `requirements.txt`)

## Setup Instructions

### 1. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 2. Setting up Redis Stack with Docker

Redis Stack is required for vector search capabilities. The easiest way to set up Redis Stack is using Docker:

```bash
# Pull the Redis Stack image
docker pull redis/redis-stack:latest

# Start Redis Stack container
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

The Redis Stack container exposes two ports:
- `6379`: Redis server port
- `8001`: RedisInsight web interface for managing and monitoring Redis

To check if the container is running:

```bash
docker ps | grep redis-stack
```

If you need to stop and restart the container:

```bash
docker stop redis-stack
docker start redis-stack
```

### 3. Verifying Redis and RediSearch

To verify that Redis and RediSearch are properly set up:

```bash
# Connect to Redis CLI in the Docker container
docker exec -it redis-stack redis-cli

# Check if RediSearch module is loaded
MODULE LIST | grep search

# List existing indices
FT._LIST
```

If you need to create the vector index manually:

```bash
docker exec -it redis-stack redis-cli FT.CREATE embedding_index ON HASH PREFIX 1 doc: SCHEMA text TEXT file TEXT page NUMERIC embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE
```

## Working with process_docs.py

The `process_docs.py` script processes PDF documents, generates vector embeddings, and stores them in Redis:

### Usage

```bash
python process_docs.py
```

When running the script, you will be asked if you want to clear existing Redis data:
- Answer `y` to clear all existing document data
- Answer `n` to preserve existing data and add new documents

### Configuration Options

You can modify these parameters in the script:
- `chunk_size`: Size of text chunks (default: 300 words)
- `overlap`: Overlap between chunks (default: 50 words)
- `EMBEDDING_MODEL`: The embedding model to use (default: "all-mpnet-base-v2" or "InstructorXL")

### Troubleshooting Redis Connectivity

If you encounter Redis connectivity issues:

1. **If Redis isn't running:**
   ```bash
   docker start redis-stack
   ```

2. **If you have both local Redis and Docker Redis running:**
   Stop the local Redis service to avoid port conflicts:
   ```bash
   brew services stop redis
   ```

3. **If RediSearch module isn't available:**
   Ensure you're using Redis Stack, not standard Redis:
   ```bash
   docker exec -it redis-stack redis-cli MODULE LIST
   ```

## Searching Documents

After processing documents, you can use the search functionality:

```bash
python search.py
```

This will start an interactive search interface where you can enter queries and get results based on semantic similarity.

## Advanced Configuration

### Using Different Embedding Models

This project supports multiple embedding models:
- "all-mpnet-base-v2" (default)
- "InstructorXL" (higher quality but slower)

To switch models, modify the `EMBEDDING_MODEL` variable in the scripts or update the configuration in `config.py`.

### Customizing Vector Search Parameters

You can customize the vector search parameters in the `create_vector_index` function in `process_docs.py`:
- `HNSW 6`: HNSW algorithm with 6 max connections per node
- `DIM 768`: Vector dimension (depends on the embedding model)
- `DISTANCE_METRIC COSINE`: Similarity metric (COSINE, L2, IP)

### Monitoring Redis

You can monitor Redis and inspect data using the RedisInsight web interface at http://localhost:8001