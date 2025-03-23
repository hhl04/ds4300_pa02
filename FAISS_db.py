import os
import fitz  # PyMuPDF
import time
import re
from langchain_huggingface import HuggingFaceEmbeddings  # 更新的导入
from langchain_community.vectorstores import FAISS  # 使用FAISS替代Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 配置
DATA_DIR = './ds4300 docs'  # PDF文件目录
FAISS_INDEX_PATH = "./faiss_index"  # FAISS索引存储路径
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 或其他您之前测试过的嵌入模型

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def extract_clean_pdf(pdf_path, remove_pgnum=True, remove_sbullets=True, clean_formatting=True, 
                     remove_whitespace=True, remove_punct=False, preserve_code=True):
    """Extract and clean text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        remove_pgnum: Whether to remove page numbers
        remove_sbullets: Whether to replace special bullet points
        clean_formatting: Whether to clean up formatting (newlines, spaces)
        remove_whitespace: Whether to trim leading/trailing whitespace
        remove_punct: Whether to remove punctuation (default: False to preserve code syntax)
        preserve_code: Whether to preserve code structure (default: True)
    
    Returns:
        Cleaned text string for the entire document
    """
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
            # Preserve code blocks (identified by indentation or common code patterns)
            # Split by lines to process each line separately
            lines = text.split('\n')
            processed_lines = []
            
            for line in lines:
                # Check if line likely contains code (simple heuristics)
                code_indicators = ['import ', 'from ', ' = ', '(', ')', 'def ', 'class ', '{}', '[]', '+=', '-=', '*=', '/=']
                is_likely_code = any(indicator in line for indicator in code_indicators)
                
                if is_likely_code and preserve_code:
                    # Minimal formatting for code - just trim excess whitespace
                    line = re.sub(r' {2,}', ' ', line)
                    processed_lines.append(line)
                else:
                    # Regular text formatting
                    line = re.sub(r' {2,}', ' ', line)
                    processed_lines.append(line)
            
            # Rejoin with proper newlines
            text = '\n'.join(processed_lines)
            
            # Carefully handle paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple blank lines to just one
            
            # Fix encoding issues
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        if remove_punct:
            # Only remove punctuation if explicitly requested
            # Be careful with code - preferably don't use this option for code-heavy documents
            if preserve_code:
                # Identify potential code blocks to preserve
                lines = text.split('\n')
                processed_lines = []
                
                for line in lines:
                    code_indicators = ['import ', 'from ', ' = ', '(', ')', 'def ', 'class ', '{}', '[]', '+=', '-=', '*=', '/=']
                    is_likely_code = any(indicator in line for indicator in code_indicators)
                    
                    if is_likely_code:
                        processed_lines.append(line)  # Preserve code lines as-is
                    else:
                        # Remove punctuation from non-code lines
                        processed_line = re.sub(r'[^\w\s]', '', line)
                        processed_lines.append(processed_line)
                
                text = '\n'.join(processed_lines)
            else:
                # Remove all punctuation
                text = re.sub(r'[^\w\s]', '', text)

        if remove_whitespace:
            text = text.strip()

        if len(text) > 3:
            cleaned_text.append(text)
    
    return "\n\n".join(cleaned_text)  # 将所有页面合并成一个文档

def process_documents():
    """处理所有PDF文件并将其存储到FAISS索引"""
    # 检查目录
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return
    
    # 获取PDF文件列表并按编号排序
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    # 按数字排序
    def extract_number(filename):
        import re
        match = re.search(r'^(\d+)', filename)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    pdf_files.sort(key=extract_number)
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    # 文本分割器 - 适合技术文档，保留较大的代码块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_documents = []
    all_metadatas = []
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            # 提取文本
            raw_text = extract_clean_pdf(pdf_path)
            
            # 分割文本为块
            texts = text_splitter.split_text(raw_text)
            
            # 创建元数据
            metadatas = [{"source": pdf_file, "chunk": i} for i in range(len(texts))]
            
            # 添加到文档列表
            all_documents.extend(texts)
            all_metadatas.extend(metadatas)
            
            print(f"  Extracted {len(texts)} chunks from {pdf_file}")
            
        except Exception as e:
            print(f"  Error processing {pdf_file}: {str(e)}")
    
    # 存储到FAISS
    print(f"\nStoring {len(all_documents)} chunks to FAISS index...")
    
    try:
        vector_store = FAISS.from_texts(
            texts=all_documents,
            embedding=embeddings,
            metadatas=all_metadatas
        )
        
        # 保存索引到本地
        vector_store.save_local(FAISS_INDEX_PATH)
        
        print(f"Successfully stored documents in FAISS index at {FAISS_INDEX_PATH}")
        return vector_store
    except Exception as e:
        print(f"Error storing documents in FAISS: {str(e)}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    vector_store = process_documents()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

# 添加更多嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embedding_models = {
    "MiniLM": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "MPNet": HuggingFaceEmbeddings(model_name="all-mpnet-base-v2"),
    "Instructor": HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
}

# 比较嵌入性能
for name, model in embedding_models.items():
    start_time = time.time()
    # 测试嵌入生成
    end_time = time.time()
    print(f"Model {name}: {end_time - start_time:.2f} seconds")

# Redis向量数据库实现
redis_url = "redis://localhost:6379"
redis_store = Redis.from_texts(
    texts=all_documents,
    embedding=embeddings,
    metadatas=all_metadatas,
    redis_url=redis_url,
    index_name="ds4300_redis"
)

# Chroma实现
chroma_store = Chroma.from_texts(
    texts=all_documents,
    embedding=embeddings,
    metadatas=all_metadatas,
    collection_name="ds4300_chroma"
)

def setup_qa_system(vector_store, llm_model="llama2"):
    # 初始化本地LLM
    llm = Ollama(model=llm_model)
    
    # 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# 测试不同LLM
llm_models = ["llama2", "mistral"]
test_questions = [
    "What is a vector database?",
    "Explain Redis replication",
    "How does MongoDB differ from relational databases?"
]

for llm_name in llm_models:
    qa = setup_qa_system(vector_store, llm_model=llm_name)
    for question in test_questions:
        result = qa({"query": question})
        print(f"Model: {llm_name}, Question: {question}")
        print(f"Answer: {result['result']}\n")

chunk_sizes = [200, 500, 1000]
overlaps = [0, 50, 100]

chunking_results = {}

for size in chunk_sizes:
    for overlap in overlaps:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # 测试分块性能和质量
        # ...

def evaluate_pipeline(embedding_model, vector_db, chunk_size, overlap, llm_model):
    # 设置管道
    # 测量性能指标
    # 评估响应质量
    return results

# 系统地测试所有组合
all_results = []
for emb in embedding_models:
    for db in vector_dbs:
        for size in chunk_sizes:
            for overlap in overlaps:
                for llm in llm_models:
                    result = evaluate_pipeline(emb, db, size, overlap, llm)
                    all_results.append(result)

# 分析结果并确定最佳管道