import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
RAW_DIR = "data/raw_documents"
PROCESSED_DIR = "data/processed/legal"

# Create the output directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# The specific files we want to process
pdf_files = ["Computer-Misuse-and-Cybercrimes-Act.pdf", "Data Protection Act.pdf"]

# We use LangChain to split the legal text so chunks overlap slightly.
# This prevents cutting a legal clause in half!
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def process_pdfs():
    for pdf_name in pdf_files:
        pdf_path = os.path.join(RAW_DIR, pdf_name)
        
        if not os.path.exists(pdf_path):
            print(f"⚠️ Warning: Could not find {pdf_path}")
            continue
            
        print(f"📄 Extracting text from {pdf_name}...")
        full_text = ""
        
        # Read the PDF page by page
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        print(f"✂️ Chunking {pdf_name}...")
        chunks = text_splitter.split_text(full_text)
        
        # Save chunks as individual txt files
        safe_name = pdf_name.replace(".pdf", "").replace(" ", "_")
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{safe_name}_chunk_{i}.txt"
            with open(os.path.join(PROCESSED_DIR, chunk_filename), "w", encoding="utf-8") as f:
                f.write(chunk)
                
        print(f"✅ Saved {len(chunks)} chunks for {pdf_name}\n")

if __name__ == "__main__":
    process_pdfs()