# FinTech RAG Chatbot

A Streamlit chatbot that answers questions over finance documents using LangChain and FAISS.

## Stack
- Streamlit
- LangChain
- FAISS
- OpenAI embeddings and chat model

## How it works
1. Load a finance text document.
2. Split into chunks.
3. Create embeddings.
4. Store vectors in FAISS.
5. Retrieve relevant chunks.
6. Generate an answer with an LLM.

## Next step
Add a finance text file in `data/finance_notes.txt`, then run:
```bash
streamlit run app.py
```