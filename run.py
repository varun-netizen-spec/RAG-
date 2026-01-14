from rag_pipeline import load_rag_pipeline

if __name__ == "__main__":
    print("\n========== RAG EXECUTION STARTED ==========")
    qa, db = load_rag_pipeline()
    
    # Example query
    query = "What domain focused on this internship?"
    result = qa.invoke(query)
    print("\n🔹 Answer:", result)
