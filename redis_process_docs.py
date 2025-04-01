import os
import fitz
import re
from config import CHUNK_SIZES, OVERLAPS

def extract_clean_pdf(
    pdf_path: str,
    remove_pgnum=True,
    remove_sbullets=True,
    clean_formatting=True,
    remove_whitespace=True,
    remove_punct=False
) -> list:
    """Extract and clean text from a PDF file, returning a list of cleaned page texts."""
    doc = fitz.open(pdf_path)
    cleaned_text = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        if remove_pgnum:
            text = re.sub(r'\n\d+\n?$', '', text)

        if remove_sbullets:
            text = text.replace("●", "-").replace("■", "-").replace("○", "-")
        
        if clean_formatting:
            text = re.sub(r'\n{2,}', '\n\n', text) 
            text = re.sub(r'\n+', ' ', text) 
            text = re.sub(r' +', ' ', text)
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        if remove_punct:
            text = re.sub(r'[^\w\s]', '', text)

        if remove_whitespace:
            text = text.strip()

        if len(text) > 3:
            cleaned_text.append(text)
    
    return cleaned_text


def split_text_into_chunks(text: str, chunk_size=300, overlap=50) -> list:
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append({
            "chunk": chunk,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "start_idx": i,
            "end_idx": min(i + chunk_size, len(words))
        })
    return chunks


def split_text_variants(text, file_name, page_num, chunk_sizes=CHUNK_SIZES, overlaps=OVERLAPS) -> list:
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


def process_pdfs(data_dir: str, verbose=True) -> list:
    all_chunks = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            cleaned_pages = extract_clean_pdf(pdf_path)

            for page_num, text in enumerate(cleaned_pages):
                split_variants = split_text_variants(
                    text, 
                    file_name=file_name,
                    page_num=page_num
                )
                all_chunks.extend(split_variants)

            if verbose:
                print(f"✅ Finished processing {file_name}")

    return all_chunks


# TESTING
if __name__ == "__main__":
    print("\n🔍 Testing extract_clean_pdf()")
    test_pdf = "./ds4300 docs/Document_DBs_&_MongoDB_Study_Guide.pdf"
    clean_pdf = extract_clean_pdf(test_pdf, remove_punct=False)
    print(clean_pdf)
