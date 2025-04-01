# 🧠 DS4300 Practical 2: RAG-Based Document Retrieval with Redis, Chroma, Milvus

This repository provides a comprehensive framework to process documents, generate embeddings, store them in vector databases (ChromaDB, Milvus, Redis), and run similarity searches for benchmarking performance across systems.

## 📁 Project Structure

```
.
├── config.py                    # Shared configuration for all systems
├── embeddings.py               # Embedding logic using HuggingFace model
├── extract_all_pdfs.py         # PDF extraction utility
├── chroma_process_docs.py      # Process and store docs in Chroma
├── chroma_search.py            # Query and search documents in Chroma
├── milvus_process_docs.py      # Process and store docs in Milvus
├── milvus_search.py            # Query and search documents in Milvus
├── redis_process_docs.py       # Process and store docs in Redis
├── redis_test.py               # Query and search documents in Redis
├── comprehensive_comparison.py # Benchmarking search results across all databases
```

## ⚙️ Setup

1. **Install Dependencies**  
   Make sure you have Python 3.8+ and install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**  
   Modify `config.py` to configure hostnames, ports, embedding models, and collection/index names for each database.

3. **Download or place PDFs**  
   Put your PDF files in a designated folder. You can change the source folder in `extract_all_pdfs.py`.

### 🧠 Ollama Setup

Install the Ollama app. Upon running the app, there is no GUI, but you can still interact using Terminal.

```bash
ollama pull mistral
ollama pull llama2
ollama pull nomic-embed-text
ollama list
```

### 🟥 Redis Setup

Install the `redis/redis-stack` image on Docker Desktop.  
When setting it up, map these ports:

- `6379:6379` (Redis)
- `8001:8001` (RedisInsight)

### 📘📙📒 Chroma Setup

Install the `chromadb` image on Docker Desktop.  
Map this port:

- `8000:8000`

### 🦅 Milvus Setup

#### For macOS:

**Step 1** – Download the Milvus Docker script:

```bash
curl -o standalone.bat https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat
```

**Step 2** – Start the Milvus container:

```bash
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.9
```

This command:
- Uses the `milvusdb/milvus:v2.3.9` image
- Automatically pulls it if not available
- Maps ports `19530` (gRPC) and `9091` (HTTP)

**Step 3** – Verify Milvus is running:

```bash
docker ps
```

#### For Windows:

Same commands as mac, with minor syntax differences:

**PowerShell / Git Bash**:

```bash
docker run -d --name milvus \`
  -p 19530:19530 \`
  -p 9091:9091 \`
  milvusdb/milvus:v2.3.9
```

**Command Prompt (CMD)**:

```cmd
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.9
```

## 🧾 How to Use

### 1. Extract and Embed Documents

```bash
python extract_all_pdfs.py
```

This extracts raw text from all PDFs in the folder.

### 2. Generate and Store Embeddings

- **Chroma**  
  ```bash
  python chroma_process_docs.py
  ```

- **Milvus**  
  ```bash
  python milvus_process_docs.py
  ```

- **Redis**  
  ```bash
  python redis_process_docs.py
  ```

### 3. Run Searches

- **Chroma Search**  
  ```bash
  python chroma_search.py
  ```

- **Milvus Search**  
  ```bash
  python milvus_search.py
  ```

- **Redis Search**  
  ```bash
  python redis_test.py
  ```

### 4. Compare Results Across All

```bash
python comprehensive_comparison.py
```

This script compares similarity search results from Chroma, Milvus, and Redis for a unified view of accuracy, speed, and relevancy.

## 🔍 Embeddings

The embeddings are handled in `embeddings.py` and use HuggingFace models (you can customize which model to use in `config.py`).

## 📊 Benchmarking Metrics

The `comprehensive_comparison.py` script prints comparison metrics like:

- Top-k result overlap
- Average search latency
- Cosine similarity scores
- Vector index health

## 🏁 Goal

Evaluate trade-offs in speed, accuracy, and usability between different vector databases for real-world document search tasks.

## ✍️ Authors: Paul Champagne, Rongxuan Zhang, Huy Le, and Yanzhen Chen.
