from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
def load_rag_pipeline():
    print("🔢 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Loading FAISS vector database...")
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("🤖 Loading LLM...")
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate.from_template(
        """
        Use the following context to answer the question.
        If you don't know the answer, say you don't know.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, db
