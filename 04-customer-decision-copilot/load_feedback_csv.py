from dotenv import load_dotenv
import os
import psycopg2
import csv

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

def load_feedback_from_csv(csv_path: str):
    conn = get_connection()
    cur = conn.cursor()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        source = row.get("source") or "Unknown"
        rating = int(row.get("rating") or 0)
        comment = row.get("comment") or ""

        if not comment:
            continue

        cur.execute(
            """
            INSERT INTO feedback (source, rating, comment)
            VALUES (%s, %s, %s);
            """,
            (source, rating, comment),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(rows)} feedback rows from CSV.")

if __name__ == "__main__":
    load_feedback_from_csv("feedback_seed.csv")