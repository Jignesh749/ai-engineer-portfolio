from dotenv import load_dotenv
import os
import psycopg2

# 1. Load environment variables from .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )

def insert_sample_data():
    conn = get_connection()
    cur = conn.cursor()

    # 2. Insert sample documents
    cur.execute(
        """
        INSERT INTO documents (title, source, content)
        VALUES (%s, %s, %s)
        """,
        (
            "Onboarding Flow Spec",
            "PRD",
            "This document describes the onboarding screens and steps for new users.",
        ),
    )

    cur.execute(
        """
        INSERT INTO documents (title, source, content)
        VALUES (%s, %s, %s)
        """,
        (
            "Pricing Policy",
            "Help Center",
            "This document explains our pricing tiers and discount rules.",
        ),
    )

    # 3. Insert sample feedback
    cur.execute(
        """
        INSERT INTO feedback (source, rating, comment)
        VALUES (%s, %s, %s)
        """,
        (
            "AppStore",
            2,
            "Onboarding is confusing and takes too many steps.",
        ),
    )

    cur.execute(
        """
        INSERT INTO feedback (source, rating, comment)
        VALUES (%s, %s, %s)
        """,
        (
            "Support",
            3,
            "I couldn't find clear information about pricing for small teams.",
        ),
    )

    # 4. Commit and close
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    insert_sample_data()
    print("Sample documents and feedback inserted.")