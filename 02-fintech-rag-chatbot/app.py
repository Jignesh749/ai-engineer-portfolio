import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="FinTech RAG Chatbot", layout="centered")
st.title("FinTech RAG Chatbot")
st.write("Ask questions about financial documents using RAG.")

DOC_PATH = "data/finance_notes.txt"

@st.cache_resource
def build_qa_chain():
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa

if os.path.exists(DOC_PATH):
    qa_chain = build_qa_chain()
    question = st.text_input("Ask a question")
    if question:
        result = qa_chain({"query": question})
        st.subheader("Answer")
        st.write(result["result"])
        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:500])
else:
    st.warning("Add a text file at data/finance_notes.txt to get started.")