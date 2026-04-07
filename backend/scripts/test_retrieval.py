import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma_db"

def test_query():
    print(">> Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name="owasp_knowledge_base")
    
    print(">> Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # A question that Pepper's trainees might ask
    query_text = "What is Broken Access Control and how do I prevent it?"
    print(f"\n?? Query: '{query_text}'")
    
    # Convert the query into a vector
    query_embedding = model.encode(query_text).tolist()
    
    # Search the database for the top 3 most relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print("\n>> Top 3 Retrieved Documents:")
    print("-" * 50)
    
    # Print out the results
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        source_file = results['metadatas'][0][i]['source']
        distance = results['distances'][0][i] # Lower distance = closer match
        
        print(f"📄 Source: {source_file} (Distance: {distance:.4f})")
        print(f"📝 Excerpt: {doc_text[:300]}...") # Print first 300 characters
        print("-" * 50)

if __name__ == "__main__":
    test_query()