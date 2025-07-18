import streamlit as st
from utils import extract_text, add_documents_to_chroma, get_top_chunks_chroma, answer_query

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title(" RAG Chatbot (with ChromaDB)")

uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    full_texts = [extract_text(f) for f in uploaded_files]
    joined_text = "\n".join(full_texts)
    chunks = [joined_text[i:i+500] for i in range(0, len(joined_text), 500)]

    add_documents_to_chroma(chunks)
    st.success("Documents added to vector store")

    query = st.text_input("Ask a question about the uploaded files:")

    if query:
        with st.spinner("Thinking..."):
            context = get_top_chunks_chroma(query)
            answer = answer_query(query, context)
        st.markdown("Answer")
        st.write(answer)
