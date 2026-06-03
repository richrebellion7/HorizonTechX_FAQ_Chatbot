from dotenv import load_dotenv
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading vector database...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("\nGITAM AI Assistant Ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=3,
        fetch_k=10
    )

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        
        content = doc.page_content[:700]
        
        context_parts.append(
            f"""
Source {i}: {source}

Content:
{content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are GITAM AI Assistant.

Answer ONLY from the provided context.

When numerical values such as fees, attendance percentages,
credit requirements, CGPA requirements, deadlines, or penalties
are present in the context, ALWAYS include the exact values.

Do not summarize away numerical information.

If the answer is not present, say:
"I could not find that information in the uploaded GITAM documents."

At the end provide:
Sources Used:
- List the sources you used in your answer, with their corresponding numbers from the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, 
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    print("\nAssistant:")
    print(response.choices[0].message.content)
    print("\n" + "-" * 60 + "\n")

    docs_with_scores = vectorstore.similarity_search_with_score(
    question,
    k=3
)

    for doc, score in docs_with_scores:
        print(score)