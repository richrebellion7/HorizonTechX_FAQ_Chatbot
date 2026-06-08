# 🎓 GITAM AI Assistant

An AI-powered university knowledge assistant that helps students find information from GITAM regulations, fee structures, syllabi, hostel policies, academic guidelines, and other official documents.

## Features

- 🔍 Semantic Search across university documents
- 🤖 AI-generated answers using Groq Llama 3.3
- 📚 Source-backed responses
- 💬 Interactive chat interface
- ⚡ Fast retrieval using FAISS
- ☁️ Streamlit deployment

![alt text](image.png)

## Tech Stack

- Python 3.11
- Streamlit
- FAISS
- HuggingFace Sentence Transformers
- Groq API
- LangChain

## Project Architecture

PDF Documents
↓
Text Chunking
↓
HuggingFace Embeddings
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
Groq LLM
↓
Student-Friendly Answer

## Example Questions

- What is the attendance requirement?
- What is the supplementary exam fee?
- What are the modules in DBMS?
- What are the hostel rules?
- How many credits are required for graduation?

## Link to Project

https://gitamh-ai-assistant.streamlit.app/

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Author

Mohammed Tazeem Wajahat

![alt text](image.png)