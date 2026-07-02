from dotenv import load_dotenv
import os
import psycopg2
from langchain_openai import OpenAIEmbeddings

# 1. Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )

def embed_documents_and_feedback():
    # 2. Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",  # good, cheap for RAG
    )

    conn = get_connection()
    cur = conn.cursor()

    # 3. Embed documents where embedding is NULL
    cur.execute("SELECT id, content FROM documents WHERE embedding IS NULL;")
    docs = cur.fetchall()

    print(f"Found {len(docs)} documents without embeddings.")

    for doc_id, content in docs:
        if not content:
            continue
        vector = embeddings.embed_query(content)  # list[float]
        # psycopg2 can store arrays directly into pgvector with %s
        cur.execute(
            "UPDATE documents SET embedding = %s WHERE id = %s;",
            (vector, doc_id),
        )

    # 4. Embed feedback where embedding is NULL
    cur.execute("SELECT id, comment FROM feedback WHERE embedding IS NULL;")
    fbs = cur.fetchall()

    print(f"Found {len(fbs)} feedback rows without embeddings.")

    for fb_id, comment in fbs:
        if not comment:
            continue
        vector = embeddings.embed_query(comment)
        cur.execute(
            "UPDATE feedback SET embedding = %s WHERE id = %s;",
            (vector, fb_id),
        )

    # 5. Commit and close
    conn.commit()
    cur.close()
    conn.close()
    print("Embeddings updated for documents and feedback.")

if __name__ == "__main__":
    embed_documents_and_feedback()