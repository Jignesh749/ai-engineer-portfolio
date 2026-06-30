import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


# ----- Load .env from repo root -----
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----- Streamlit UI setup -----
st.set_page_config(page_title="FinTech RAG Chatbot", layout="centered")
st.title("FinTech RAG Chatbot")
st.write("Upload a PDF or TXT and ask questions using retrieval-augmented generation (RAG).")

uploaded_file = st.file_uploader(
    "Upload a financial document (PDF or TXT):",
    type=["pdf", "txt"],
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {"role": "user"/"assistant", "content": str}


def build_qa_chain_from_docs(docs):
    """Build a RetrievalQA chain from a list of LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    return qa


def load_uploaded_file(uploaded_file):
    """Convert the uploaded Streamlit file into LangChain Documents."""
    if uploaded_file is None:
        return None

    fname = uploaded_file.name.lower()

    # Handle TXT
    if fname.endswith(".txt"):
        temp_path = Path("temp_uploaded.txt")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = TextLoader(str(temp_path), encoding="utf-8")
        docs = loader.load()

        temp_path.unlink(missing_ok=True)
        return docs

    # Handle PDF
    if fname.endswith(".pdf"):
        temp_path = Path("temp_uploaded.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(str(temp_path))
        docs = loader.load()

        temp_path.unlink(missing_ok=True)
        return docs

    # Fallback
    return None


# ----- App logic -----
if uploaded_file is None:
    st.info("Upload a PDF or TXT file to get started.")
else:
    docs = load_uploaded_file(uploaded_file)
    if not docs:
        st.error("Could not load the uploaded file. Please try a different PDF or TXT.")
    else:
        # Build RAG chain from uploaded docs
        qa_chain = build_qa_chain_from_docs(docs)

        # Show conversation history
        st.subheader("Conversation history")
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

        # Current question input
        question = st.text_input("Ask a question about the uploaded document:")

        if question:
            # Store user question
            st.session_state["messages"].append(
                {"role": "user", "content": question}
            )

            # Run RAG chain
            result = qa_chain({"query": question})
            answer = result["result"]

            # Store assistant answer
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            # Display current answer
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for i, doc in enumerate(result["source_documents"], start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:1000])