import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

index = faiss.read_index("index.faiss")
with open("chunks.json", "r") as f:
    chunks = json.load(f)
    
model  = SentenceTransformer('all-MiniLM-L6-v2')
client= Groq()

def ask(query, top_k = 3):
    q_embedding = model.encode([query])
    _, indices = index.search(np.array(q_embedding, dtype="float32"), top_k)
    context = " ".join([chunks[i] for i in indices[0]])
    
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {
                "role" : "system",
                "content" : "You are a helpful assistant that answers questions based on the provided context." + context       
            },
            {
                "role" : "user",
                "content" : query
            }
        ]
    )
    return response.choices[0].message.content
    
print("Welcome to RAG : Enter your query")
while(True):
    query = input("Enter your query: ").strip()
    print("\nAnswer : ", ask(query), "\n\n")