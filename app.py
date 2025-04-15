import requests
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load pre-trained AI model for generation
qa = pipeline("text-generation", model="gpt2")

# Replace with your SerpAPI key
SERPAPI_KEY = "YOUR_SERPAPI_KEY"

class Question(BaseModel):
    query: str

def search_web(query):
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google"
    }
    r = requests.get("https://serpapi.com/search", params=params)
    data = r.json()
    return data["organic_results"][0]["snippet"]

@app.post("/ask")
def ask_question(data: Question):
    try:
        context = search_web(data.query)
        prompt = f"Context: {context}\nQuestion: {data.query}\nAnswer:"
        answer = qa(prompt, max_length=150, pad_token_id=50256)[0]["generated_text"]
        return {"answer": answer[len(prompt):].strip()}
    except Exception as e:
        return {"error": str(e)}
