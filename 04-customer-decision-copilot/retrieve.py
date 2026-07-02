from dotenv import load_dotenv
import os
import psycopg2
from langchain_openai import OpenAIEmbeddings

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

def search_feedback(query: str, k: int = 5):
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )

    # 1. Turn query into an embedding
    query_vec = embeddings.embed_query(query)

    conn = get_connection()
    cur = conn.cursor()

    # 2. Find nearest feedback comments by vector similarity
    cur.execute(
        """
        SELECT id, source, rating, comment
        FROM feedback
        ORDER BY embedding <-> CAST(%s AS vector)
        LIMIT %s;
        """,
        (query_vec, k),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

if __name__ == "__main__":
    results = search_feedback("What are users complaining about onboarding?")
    for r in results:
        print(r)