# FinTech RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets users upload their own finance documents (TXT or PDF) and ask natural-language questions about them. Instead of manually reading long notes or reports, the chatbot retrieves relevant passages and generates grounded answers in seconds.

The app is built with:

- Streamlit for the web UI.
- LangChain for the RAG pipeline.
- FAISS for fast vector search.
- OpenAI API for embeddings and answer generation.

It is designed as an AI engineer portfolio project focused on financial use cases like credit scores, interest rates, loan terms, and credit risk.

---

## Features

- **Document upload**: Users can upload TXT or PDF files containing finance content (notes, reports, articles).
- **RAG pipeline**:
  - Load the uploaded document into LangChain `Document` objects.
  - Split text into chunks using `RecursiveCharacterTextSplitter`.
  - Create embeddings with `OpenAIEmbeddings`.
  - Store and search chunks using FAISS.
  - Answer questions with a `RetrievalQA` chain that uses the retrieved chunks as context.
- **Chat interface**:
  - Streamlit-based interface with a text input box for questions.
  - Conversation history stored in `st.session_state` so users can see previous questions and answers.
- **Grounded answers**:
  - For each question, the app shows both the generated answer and the source text snippets used, helping demonstrate RAG behavior.
- **Secure API key management**:
  - OpenAI API key stored in a `.env` file at the repo root.
  - `.env` is ignored by Git via `.gitignore`, so secrets are never committed to GitHub.

---

## Tech Stack

- **Language**: Python
- **UI**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: FAISS (CPU)
- **LLM & Embeddings**: OpenAI (via `langchain-openai`)
- **Config**: `python-dotenv` for loading `.env`

---

## Project Structure

Inside the main portfolio repo:

```text
ai-engineer-portfolio/
├── 01-credit-risk-model/
├── 02-fintech-rag-chatbot/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── data/
│       └── sample_finance_notes.txt  (example document)
└── 03-trading-signal-engine/
```

At the repo root:

- `.env` – contains `OPENAI_API_KEY=...` (not committed).
- `.gitignore` – includes `.env` and `.DS_Store`.

---

## How It Works

1. **Environment and key loading**

   - The app uses `dotenv` to load `.env` from the repo root:
     - `OPENAI_API_KEY` is read via `os.getenv("OPENAI_API_KEY")`.
   - The key is passed explicitly to `OpenAIEmbeddings` and `ChatOpenAI` so the RAG pipeline can call the OpenAI API.

2. **Document upload and loading**

   - Users upload a `.txt` or `.pdf` file via `st.file_uploader`.
   - The uploaded file is saved temporarily and loaded with:
     - `TextLoader` for `.txt`.
     - `PyPDFLoader` for `.pdf`.
   - The loader returns a list of LangChain `Document` objects.

3. **Chunking, embeddings, and vector store**

   - Text chunks are created using `RecursiveCharacterTextSplitter`.
   - Embeddings are generated with `OpenAIEmbeddings`.
   - A FAISS vector store is built with `FAISS.from_documents(chunks, embeddings)`.

4. **Question answering (RAG)**

   - A `RetrievalQA` chain is built using:
     - `ChatOpenAI` (e.g. `gpt-4o-mini`) for answer generation.
     - The FAISS retriever for semantic search.
   - For each user question:
     - The retriever finds the top-k relevant chunks.
     - The LLM generates an answer grounded in those chunks.
     - The app displays both the answer and the retrieved source snippets.

5. **Chat history**

   - Questions and answers are stored in `st.session_state["messages"]`.
   - The conversation history is displayed above the input box with “You:” and “Assistant:” messages.

---

## Setup and Usage

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/ai-engineer-portfolio.git
   cd ai-engineer-portfolio/02-fintech-rag-chatbot
   ```

2. **Create `.env` at the repo root**

   At `ai-engineer-portfolio/.env`:

   ```text
   OPENAI_API_KEY=sk-...your-openai-key...
   ```

   Ensure `.env` is listed in `.gitignore` so it is never committed.

3. **Install dependencies**

   From `02-fintech-rag-chatbot`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

5. **Upload a document and ask questions**

   - Upload a finance-related TXT or PDF (for example, `sample_finance_notes.txt` containing notes about credit scores, interest rates, and risk management).
   - Type questions such as:
     - “How do credit scores affect interest rates?”
     - “Explain loan terms and their impact on monthly payments.”
     - “What tools does a bank use to manage credit risk?”

---

## What This Project Demonstrates

This project demonstrates:

- Practical use of RAG (Retrieval-Augmented Generation) for document Q&A.
- Integration of Streamlit, LangChain, FAISS, and OpenAI.
- Secure API key management using `.env` and `.gitignore`.
- Handling file uploads and converting them into LangChain documents.
- Basic chat history with `st.session_state`.

It is intended as a portfolio example for AI engineer / data scientist roles with a focus on FinTech and applied LLMs.

---

## Possible Extensions

Ideas for future improvements:

- Support multiple uploaded documents and multi-file search.
- Add token usage and approximate cost per answer.
- Add richer chat UI (e.g., `st.chat_message`, avatars, timestamps).
- Plug in other model providers (e.g., Gemini, Groq, or local LLMs) to reduce cost.
- Deploy the app on Streamlit Cloud or another hosting platform.
