from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Customer Decision Copilot")

# Serve static files if needed (optional for now)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

    query_vec = embeddings.embed_query(query)

    conn = get_connection()
    cur = conn.cursor()

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

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    feedback: list[dict]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1. Retrieve relevant feedback
    rows = search_feedback(req.question, k=5)
    if not rows:
        # No feedback found for this question
        return AskResponse(
            answer=(
                "I couldn't find any customer feedback relevant to this question. "
                "Try rephrasing it or checking that feedback has been loaded into the system."
            ),
            feedback=[],
        )

    
    # 2. Build context for the LLM
    feedback_snippets = []
    for (fb_id, source, rating, comment) in rows:
        feedback_snippets.append(
            f"[{source} | rating={rating}] {comment}"
        )

    context_text = "\n".join(feedback_snippets)

    # 3. Call gpt-4o-mini with question + context
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2,
    )

    prompt = (
        "You are a product insights assistant. "
        "Use the following customer feedback to answer the question.\n\n"
        "Customer feedback:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{req.question}\n\n"
        "Answer with a concise summary of what customers are saying, "
        "and mention specific themes or pain points."
    )

    response = llm.invoke(prompt)
    answer_text = response.content

    # 4. Return answer + raw feedback
    feedback_list = [
        {"id": fb_id, "source": source, "rating": rating, "comment": comment}
        for (fb_id, source, rating, comment) in rows
    ]

    return AskResponse(answer=answer_text, feedback=feedback_list)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Customer Decision Copilot</title>
      <style>
        body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
        textarea { width: 100%; height: 80px; }
        button { margin-top: 10px; padding: 8px 16px; }
        .answer { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
        .feedback-item { margin-top: 8px; font-size: 0.9em; color: #555; }
      </style>
    </head>
    <body>
      <h1>Customer Decision Copilot</h1>
      <p>Ask a question about what customers are saying, and see insights + raw feedback.</p>
      <textarea id="question" placeholder="e.g. What are users complaining about onboarding?"></textarea><br/>
      <button onclick="submitQuestion()">Ask</button>

      <div class="answer" id="answer"></div>
      <div id="feedback"></div>

      <script>
        async function submitQuestion() {
          const question = document.getElementById("question").value;
          const answerDiv = document.getElementById("answer");
          const feedbackDiv = document.getElementById("feedback");

          answerDiv.innerHTML = "Thinking...";
          feedbackDiv.innerHTML = "";

          try {
            const res = await fetch("/ask", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ question }),
            });

            if (!res.ok) {
              answerDiv.innerHTML = "Error: " + res.status;
              return;
            }

            const data = await res.json();
            answerDiv.innerHTML = "<strong>Answer:</strong><br/>" + data.answer;

            let fbHtml = "<h3>Underlying feedback</h3>";
            for (const fb of data.feedback) {
              fbHtml += `<div class="feedback-item">
                [${fb.source} | rating=${fb.rating}] ${fb.comment}
              </div>`;
            }
            feedbackDiv.innerHTML = fbHtml;
          } catch (err) {
            answerDiv.innerHTML = "Error: " + err;
          }
        }
      </script>
    </body>
    </html>
    """