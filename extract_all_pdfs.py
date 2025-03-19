import os
import fitz
import re
import time
import nltk
from nltk.corpus import stopwords
import sys

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Use relative path
DATA_DIR = './ds4300 docs'
OUTPUT_DIR = './extracted_text'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
        List of cleaned text strings, one per page
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
    
    return cleaned_text

def preprocess_text(text, strategy='standard'):
    """Apply different preprocessing strategies to text based on assignment requirements.
    
    Args:
        text: The input text to preprocess
        strategy: One of 'standard', 'minimal', 'aggressive', or 'code_preserve'
    
    Returns:
        Processed text
    """
    if strategy == 'minimal':
        # Minimal processing - just fix basic issues
        processed = text.strip()
        processed = re.sub(r'\s+', ' ', processed)  # Normalize whitespace
        return processed
        
    elif strategy == 'standard':
        # Standard processing - clean whitespace, preserve punctuation
        processed = text.strip()
        processed = re.sub(r'\s+', ' ', processed)  # Normalize whitespace
        # Fix common encoding issues
        processed = processed.encode('utf-8', 'ignore').decode('utf-8')
        return processed
        
    elif strategy == 'aggressive':
        # Aggressive processing - remove punctuation, stopwords, lowercase
        processed = text.lower()
        processed = re.sub(r'[^\w\s]', '', processed)  # Remove punctuation
        processed = re.sub(r'\s+', ' ', processed)  # Normalize whitespace
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = processed.split()
        processed = ' '.join([word for word in words if word not in stop_words])
        
        return processed
        
    elif strategy == 'code_preserve':
        # Processing that preserves code structure
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Check if line likely contains code
            code_indicators = ['import ', 'from ', ' = ', '(', ')', 'def ', 'class ', '{}', '[]']
            is_likely_code = any(indicator in line for indicator in code_indicators)
            
            if is_likely_code:
                # Preserve code lines with minimal changes
                processed_lines.append(line.strip())
            else:
                # Process normal text lines
                processed_line = line.strip()
                processed_line = re.sub(r'\s+', ' ', processed_line)
                processed_lines.append(processed_line)
                
        return '\n'.join(processed_lines)
        
    else:
        # Default fallback
        return text.strip()

def process_all_pdfs():
    """Process all PDF files in the data directory and save results to text files."""
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"Error: No PDF files found in {DATA_DIR}.")
        return
    
    # Sort PDF files numerically if they contain numbers (e.g., "01 - Introduction.pdf")
    def extract_number(filename):
        # Try to extract number from the beginning of the filename
        match = re.search(r'^(\d+)', filename)
        if match:
            return int(match.group(1))
        # If no number at beginning, check anywhere in the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        # If no number found, sort alphabetically
        return 0
    
    # Sort files based on numeric values in their names
    pdf_files.sort(key=extract_number)
    
    print(f"Found {len(pdf_files)} PDF files.")
    print("Files will be processed in the following order:")
    for i, pdf_file in enumerate(pdf_files):
        print(f"  {i+1}. {pdf_file}")
    
    # Create separate output files for each strategy
    strategies = ['minimal', 'standard', 'aggressive', 'code_preserve']
    output_files = {}
    
    for strategy in strategies:
        output_path = os.path.join(OUTPUT_DIR, f"{strategy}_processed.txt")
        output_files[strategy] = open(output_path, 'w', encoding='utf-8')
        output_files[strategy].write(f"# Documents processed with {strategy} strategy\n\n")
    
    # Process all PDF files
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            # Extract text
            cleaned_pages = extract_clean_pdf(pdf_path)
            full_document = "\n\n".join(cleaned_pages)
            
            # Apply preprocessing for each strategy and write to corresponding file
            for strategy in strategies:
                processed_text = preprocess_text(full_document, strategy=strategy)
                
                output_files[strategy].write(f"\n\n{'='*40}\n")
                output_files[strategy].write(f"Document: {pdf_file}\n")
                output_files[strategy].write(f"{'='*40}\n\n")
                output_files[strategy].write(processed_text)
                output_files[strategy].write("\n\n")
                
                # Ensure immediate write to file
                output_files[strategy].flush()
            
            print(f"  Finished processing {pdf_file}.")
            
        except Exception as e:
            print(f"  Error processing {pdf_file}: {str(e)}")
    
    # Close all output files
    for file in output_files.values():
        file.close()
    
    print("\nProcessing complete!")
    print(f"Text has been saved to the following files in the {OUTPUT_DIR} directory:")
    for strategy in strategies:
        print(f"  - {strategy}_processed.txt")

if __name__ == "__main__":
    start_time = time.time()
    print("Starting to process all PDF files...\n")
    process_all_pdfs()
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds") 