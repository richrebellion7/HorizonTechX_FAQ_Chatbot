import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="GITAM AI Assistant",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_vectorstore():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

vectorstore = load_vectorstore()
st.title("🎓 GITAM AI Assistant")
st.markdown(
    """
Ask questions about:

- Attendance
- Fees
- Hostel
- Syllabus
- Academic Regulations
- Examination Policies
"""
)

with st.sidebar:
    st.header("Suggested Questions")
    st.markdown("""
- What is attendance requirement?
- What is supplementary exam fee?
- What is hostel fee?
- What is ragging?
- How many credits are required?
- What are the modules in DBMS?
""")
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a GITAM-related question...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = vectorstore.max_marginal_relevance_search(question, k=3, fetch_k=10)
            
            st.write(f"DEBUG: Found {len(docs)} documents.")
            for d in docs:
                st.write(f"DEBUG: Preview: {d.page_content[:100]}...")

            context_parts = []
            sources = set()

            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                sources.add(source)
                context_parts.append(
                    f"""
Source: {source}

Content:
{doc.page_content[:700]}
"""
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are GITAM AI Assistant.

Answer ONLY using the provided context.

Rules:

1. Give a direct answer.
2. Use exact numbers if present.
3. Keep answers concise.
4. Explain in student-friendly language.
5. If information is unavailable, say:
   "I could not find that information in the uploaded GITAM documents."

Context:
{context}

Question:
{question}

Answer:
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_completion_tokens=800
            )

            answer = response.choices[0].message.content

            st.markdown(answer)
            with st.expander("📚 Sources Used"):
                for source in sorted(sources):
                    st.write(f"📄 {source}")

            source_list_md = "\n".join([f"- 📄 {s}" for s in sorted(sources)])
            response_text = f"{answer}\n\n**Sources Used:**\n{source_list_md}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )