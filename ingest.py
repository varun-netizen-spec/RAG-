from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

print("\n========== DOCUMENT INGESTION STARTED ==========\n")

print("📄 Loading PDF...")
loader = PyPDFLoader("C:\\Users\\varun\\OneDrive\\Desktop\\RAG\\data\\Internship_Project_Report_Elevate_Labs.pdf")
documents = loader.load()
print(f"✅ Pages loaded: {len(documents)}")

print("\n✂️ Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

print("\n🔢 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\n📦 Saving FAISS index...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print("\n🎉 INGESTION COMPLETED SUCCESSFULLY!\n")

