import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.markdown import Markdown

console = Console()

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Loading FAISS index...")
index = faiss.read_index("studentlife_profiles.faiss")

print("Loading metadata...")
with open("studentlife_profiles.json", "r") as f:
    metadata = json.load(f)

print("Loading embeddings model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

model = genai.GenerativeModel("gemini-2.5-flash")

def retrieve_chunks(query, k=5):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [metadata[i]["profile"] for i in indices[0]]

def answer_question(query, k=5):
    top_chunks = retrieve_chunks(query, k)
    context = "\n\n".join(top_chunks)
    print(context)

    prompt = f"""
    
    You are an expert college major advisor.
    
    Here are some college students' profiles from Dartmouth with information like their grades, responses to surveys regarding their personality and well-being (stess, etc.), and social life.
    Based on this help answer the student's question and suggest a good college major or give advice on how they can choose one.

Similar Dartmouth Students' Profiles:
{context}

User question: {query}.
"""

    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask a question about choosing a major (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = answer_question(query)
        console.print(Markdown(f"\n💡 Answer:\n{answer}\n"))
