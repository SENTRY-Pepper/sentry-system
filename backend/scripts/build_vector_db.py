import os
import chromadb
from sentence_transformers import SentenceTransformer

# Paths relative to the 'backend' dir
BASE_PROCESSED_DIR = "data/processed"
DB_DIR = "data/chroma_db"

def build_database():
    print("> Initializing ChromaDB Persistent Client...")
    # Creates a local SQLite-based vector DB in the DB_DIR folder
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Create a collection (like a table in a traditional DB)
    collection = client.get_or_create_collection(name="owasp_knowledge_base")
    
    print(">> Loading Embedding Model: all-MiniLM-L6-v2...")
    print("(This might take a minute on the first run as it downloads the model)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = []
    metadatas = []
    ids = []
    
    print(f"> Reading chunked files from all subfolders in {BASE_PROCESSED_DIR}...")
    
    # Walk through the directory tree (this will catch 'owasp' and 'legal' folders)
    for root, _, files in os.walk(BASE_PROCESSED_DIR):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                documents.append(content)
                metadatas.append({"source": filename}) 
                ids.append(filename)

    if not documents:
        print("!! No text files found to embed. Check your processed data folder.")
        return

    print(f"?> Embedding {len(documents)} documents. This requires some CPU power...")
    # Convert the text into numbers (vectors)
    embeddings = model.encode(documents).tolist()
    
    print(">>> Saving embeddings to ChromaDB...")
    # Add everything to the database
    collection.upsert(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f">> Success! Vector database securely populated at {DB_DIR}")

if __name__ == "__main__":
    build_database()