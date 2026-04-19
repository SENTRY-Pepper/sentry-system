import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os

class SentryRAG:
    def __init__(self, db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
        # 1. Load the Embedding Model (Local)
        self.model = SentenceTransformer(model_name)
        
        # 2. Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("cyber_laws")
        
        # 3. Set OpenAI Key
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def query(self, user_input: str, threshold: float = 0.7):
        # A. Embed the user's question
        query_vector = self.model.encode(user_input).tolist()

        # B. Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        # C. Check Similarity (Grounding Gate)
        # Note: ChromaDB distances are often L2; a smaller distance means higher similarity.
        # If your distance is too high (e.g., > 1.0), the content is irrelevant.
        if not results['documents'][0] or results['distances'][0][0] > 1.0:
            return "I'm sorry, I don't have verified information on that in the Kenyan Cybersecurity framework.", "None", 0.0, True

        # D. Construct the Grounded Prompt
        context = "\n".join(results['documents'][0])
        sources = ", ".join(set([m.get("source", "Unknown") for m in results['metadatas'][0]]))
        
        prompt = f"""
        You are SENTRY, a cybersecurity trainer robot. Use ONLY the following legal context to answer.
        If the answer isn't in the context, say you don't know.
        
        Context: {context}
        User Question: {user_input}
        """

        # E. Get Response from GPT-4o-mini
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a precise cybersecurity assistant."},
                      {"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        score = 1 - results['distances'][0][0] # Simple conversion to a 'similarity score'
        
        return answer, sources, score, False