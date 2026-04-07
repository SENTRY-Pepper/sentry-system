import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FOLDER = "data/raw_documents/owasp"
OUTPUT_FOLDER = "data/processed/owasp"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_markdown(text):
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r'`.*?`', '', text)

    # Remove excessive symbols
    text = re.sub(r'[#>*\-]', '', text)

    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".md"):
        file_path = os.path.join(INPUT_FOLDER, file)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = clean_markdown(raw_text)

        chunks = splitter.split_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            out_file = f"{file.replace('.md','')}_chunk_{i}.txt"
            out_path = os.path.join(OUTPUT_FOLDER, out_file)

            with open(out_path, "w", encoding="utf-8") as out:
                out.write(chunk)

print(">> OWASP preprocessing complete.")