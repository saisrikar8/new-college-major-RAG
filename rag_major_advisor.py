import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class RAGMajorAdvisor:
    def __init__(self, data_path="student_data_extended - Training.csv"):
        self.df = pd.read_csv(data_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = genai.GenerativeModel("gemini-2.5-flash")
        self.index = faiss.read_index("studentlife_profiles.faiss")
        self.metadata = []
        #self._build_vector_db()

    def _create_student_profile(self, uid):
        """Creates a text chunk for a single student."""
        try:
            student_data = self.df[self.df['uid'] == uid].iloc[0].to_dict()
            profile_lines = [f"Student {uid} Profile:"]
            # exclude uid from  profile text itself but keep it for metadata
            for key, value in student_data.items():
                if key != 'uid' and pd.notna(value):
                    profile_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            return "\n".join(profile_lines)
        except IndexError:
            return None

    def _build_vector_db(self):
        """Builds the FAISS vector database and metadata."""
        print("Building vector database...")
        profile_texts = []
        for uid in self.df['uid'].unique():
            profile_text = self._create_student_profile(uid)
            if profile_text:
                student_gpa = self.df[self.df['uid'] == uid]['gpa all'].iloc[0]
                self.metadata.append({"uid": uid, "profile": profile_text, "gpa": float(student_gpa)})
                profile_texts.append(profile_text)

        print(f"Generating embeddings for {len(profile_texts)} profiles...")
        embeddings = self.embedder.encode(profile_texts, convert_to_numpy=True, show_progress_bar=True)
        
        # part for FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_chunks(self, query, k=5):
        """Retrieves the top-k most similar profiles from the vector DB."""
        query_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k)
        return [self.metadata[i] for i in indices[0]]

    def answer_question(self, user_profile, k=5):
        """Answers a question using the RAG pipeline."""
        print("Retrieving similar student profiles...")
        retrieved_docs = self.retrieve_chunks(user_profile, k)
        context = "\n\n---\n\n".join([doc["profile"] for doc in retrieved_docs])

        prompt = f"""
        You are an expert college major advisor at Dartmouth.
        
        A student has provided their profile. Based on their information, and by learning from the profiles of similar past students, suggest a suitable college major or provide actionable advice on how they can choose one.

        **Student's Profile:**
        {user_profile}

        **Similar Dartmouth Students' Profiles for Context:**
        {context}

        **Your Advice:**
        """
        
        print("Generating advice with the language model...")
        response = self.llm.generate_content(prompt)
        return response.text

# main part
if __name__ == "__main__":
    advisor = RAGMajorAdvisor()

    # example: A new student's profile to get advice for
    new_student_query = """
    I see myself as someone who is original and comes up with new ideas. 
    I'm curious about many different things and am a deep thinker. 
    However, I can be somewhat careless and get tense under pressure. 
    My grades are okay, around a 3.4 GPA. What major would fit me?
    """

    advice = advisor.answer_question(new_student_query)
    
    print("\n\n--- ADVICE FOR STUDENT ---")
    print(advice)
