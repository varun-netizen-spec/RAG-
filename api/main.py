from fastapi import FastAPI
from rag.retriever import retrieve_context
from rag.generator import generate_answer

app = FastAPI(title="AgroStock RAG Market Advisor")

@app.post("/ask")
def ask_ai(query: str):
    context = retrieve_context(query)
    answer = generate_answer(context, query)
    return {
        "query": query,
        "answer": answer
    }
