import json
import random
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

with open("college_major_metadata.json", "r") as f:
    data = json.load(f)

sampled_chunks = [entry["chunk"] for entry in random.sample(data, min(100, len(data)))]
combined_text = "\n\n".join(sampled_chunks)

prompt = f"""
You are a college advisor analyzing public Reddit discussions to extract patterns.
Here is a selection of real comments and posts from students discussing their college major decisions:

{combined_text}

Based on this, summarize the main themes, concerns, and motivations that students express when selecting a college major.
Provide a thoughtful and organized summary. Avoid repeating content verbatim.
"""

response = model.generate_content(prompt)
summary = response.text

print("🔍 Summary of Reddit Data:\n")
print(summary)

with open("summary.txt", "w") as f:
    f.write(summary)
