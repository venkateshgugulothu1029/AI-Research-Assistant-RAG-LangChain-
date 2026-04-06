import streamlit as st
import os
from dotenv import load_dotenv

from utils.loader import load_documents
from utils.splitter import split_documents
from utils.embeddings import create_vector_store
from utils.qa_chain import build_qa_chain

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant (RAG + LangChain)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        docs = load_documents("temp.pdf")
        split_docs = split_documents(docs)
        vectorstore = create_vector_store(split_docs)
        qa_chain = build_qa_chain(vectorstore)

    st.success("Document processed successfully!")

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Generating answer..."):
            response = qa_chain.run(query)

        st.subheader("Answer:")
        st.write(response)
