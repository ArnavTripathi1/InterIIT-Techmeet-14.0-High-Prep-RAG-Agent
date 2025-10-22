import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

def query_rag(question, retrieved):
    
    if not retrieved:
        return "I’m sorry, I could not find relevant information in the documents."

    context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)])

    prompt = (
        "Answer the question based only on the following context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer briefly, citing passage numbers in brackets like [1] or [2]."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "content": prompt}]
    )

    return response.text