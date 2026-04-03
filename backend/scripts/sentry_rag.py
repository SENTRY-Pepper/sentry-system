import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
DB_DIR = "data/chroma_db"

class SentryBrain:
    def __init__(self):
        print("🧠 Waking up SENTRY...")
        # 1. Connect to Memory (ChromaDB)
        self.db_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.db_client.get_collection(name="owasp_knowledge_base")
        
        # 2. Load Embedding Model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # System Prompt enforces SENTRY's persona and strict grounding rules
        self.system_prompt = """
        You are SENTRY, a cybersecurity training robot for SMEs in Kenya. 
        Your goal is to educate employees on cybersecurity threats safely and accurately.
        
        RULES:
        1. You MUST answer the user's question using ONLY the provided context.
        2. If the answer is not in the context, say "I don't have enough verified information to answer that." Do NOT guess.
        3. Keep your answers conversational, concise, and easy to understand.
        4. ALWAYS cite the source document name at the end of your response (e.g., "[Source: Data Protection Act]").
        """

    def ask(self, user_query):
        # Step 1: Embed the user's question
        query_embedding = self.embedding_model.encode(user_query).tolist()
        
        # Step 2: Retrieve relevant documents
        print(f"\n🔍 Searching memory for: '{user_query}'")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Step 3: Format the context
        context_texts = []
        sources = set()
        
        for i in range(len(results['documents'][0])):
            context_texts.append(results['documents'][0][i])
            sources.add(results['metadatas'][0][i]['source'])
            
        context_block = "\n\n---\n\n".join(context_texts)
        
        # Step 4: Construct the prompt for OpenAI
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context Information:\n{context_block}\n\nUser Question: {user_query}"}
        ]
        
        # Step 5: Generate the response
        print("💬 Generating grounded response...")
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Using a smaller model to save your budget
            messages=messages,
            temperature=0.2 # Low temperature reduces hallucination
        )
        
        return response.choices[0].message.content

# Test the Engine
if __name__ == "__main__":
    sentry = SentryBrain()
    
    # Let's test a question that should pull from your newly added legal PDFs
    test_question = "What happens if I unlawfully access a computer system according to Kenyan law?"
    
    answer = sentry.ask(test_question)
    
    print("\n" + "="*50)
    print(f"🤖 SENTRY SAYS:\n{answer}")
    print("="*50)