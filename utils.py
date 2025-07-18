import os
import fitz  
from docx import Document
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)


chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="rag_collection",
    embedding_function=embedding_function
)



def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""


def add_documents_to_chroma(chunks):
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def get_top_chunks_chroma(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]



def answer_query(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Answer the following question based on the context below.\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n\n"
        f"### Answer:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()
