import fitz
import re

def extract_clean_pdf(pdf_path, remove_pgnum=True, remove_sbullets=True, clean_formatting=True, remove_whitespace=True, remove_punct=True):
    doc = fitz.open(pdf_path)
    extracted_text = {}

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
            extracted_text[page_num + 1] = text

    return extracted_text

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def split_text_variants(text, file_name, page_num, chunk_sizes, overlaps):
    split_variants = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            chunks = split_text_into_chunks(text, chunk_size, overlap)
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
