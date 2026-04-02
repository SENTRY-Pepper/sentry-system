import os
import chromadb
from sentence_transformers import SentenceTransformer

# Define paths relative to the 'backend' directory
PROCESSED_DIR = "data/processed/owasp"
DB_DIR = "data/chroma_db"

def build_database():
    print("🚀 Initializing ChromaDB Persistent Client...")
    # This creates a local SQLite-based vector database in the DB_DIR folder
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Create a collection (think of this as a table in a traditional DB)
    collection = client.get_or_create_collection(name="owasp_knowledge_base")
    
    print("🧠 Loading Embedding Model: all-MiniLM-L6-v2...")
    print("(This might take a minute on the first run as it downloads the model)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = []
    metadatas = []
    ids = []
    
    print(f"📂 Reading chunked files from {PROCESSED_DIR}...")
    
    # Ensure the directory exists to avoid errors
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ Error: Could not find {PROCESSED_DIR}")
        return

    # Loop through all the processed text chunks
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            documents.append(content)
            # Store the source file name. The LLM will use this to say "According to A01..."
            metadatas.append({"source": filename}) 
            ids.append(filename) # Using filename as a unique ID for the chunk

    if not documents:
        print("⚠️ No text files found to embed. Check your processed data folder.")
        return

    print(f"⚙️ Embedding {len(documents)} documents. This requires some CPU power...")
    # Convert the text into numbers (vectors)
    embeddings = model.encode(documents).tolist()
    
    print("💾 Saving embeddings to ChromaDB...")
    # Add everything to the database
    collection.upsert(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Success! Vector database securely populated at {DB_DIR}")

if __name__ == "__main__":
    build_database()