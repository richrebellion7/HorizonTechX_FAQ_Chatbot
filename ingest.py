from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

print("=" * 50)
print("GITAM AI Assistant - Knowledge Base Builder")
print("=" * 50)

print("\nLoading PDFs...")

loader = PyPDFDirectoryLoader("data")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

print("\nSplitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nGenerating embeddings and creating FAISS database...")

start_time = time.time()

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

elapsed = time.time() - start_time

print(f"\nEmbedding generation completed in {elapsed:.2f} seconds")

print("\nSaving vector database...")

vectorstore.save_local("vectorstore")

print("\nDone!")
print("Vector database saved to vectorstore/")
print(f"Total documents processed: {len(documents)}")
print(f"Total chunks indexed: {len(chunks)}")