import os
import uuid

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from matplotlib.figure import Figure
from pydantic import BaseModel
from textblob import TextBlob

# Ensure plot directory exists
OS_PLOT_DIR = "plots"
os.makedirs(OS_PLOT_DIR, exist_ok=True)

app = FastAPI(title="Narrative Flow AI", version="1.0.0")


def ensure_nltk_resources() -> None:
    """Ensure the sentence tokenizer data required by TextBlob is available."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass

    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)


@app.on_event("startup")
def startup_event() -> None:
    ensure_nltk_resources()


# Fix 1: Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix 3: Mount static files directory for plot serving
app.mount("/plots", StaticFiles(directory=OS_PLOT_DIR), name="plots")


class TextRequest(BaseModel):
    text: str


def analyze_sentiment(text: str) -> list[float]:
    ensure_nltk_resources()
    blob = TextBlob(text)
    return [sentence.sentiment.polarity for sentence in blob.sentences]


def create_sentiment_plot(polarities: list[float], filename: str) -> str:
    # Fix 2: Object-oriented Matplotlib to avoid thread-safety / memory leak issues
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.plot(polarities, marker="o", linestyle="-", color="b")
    ax.set_title("Emotional Arc of the Text", fontsize=16)
    ax.set_xlabel("Sentence Index", fontsize=12)
    ax.set_ylabel("Sentiment Polarity (-1 = Negative, 1 = Positive)", fontsize=12)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.grid(True)
    ax.set_ylim(-1.1, 1.1)

    plot_path = os.path.join(OS_PLOT_DIR, filename)
    fig.savefig(plot_path, format="png")
    return plot_path


def get_structural_feedback(polarities: list[float]) -> list[str]:
    feedback = []
    if len(polarities) > 2:
        max_pol, min_pol = max(polarities), min(polarities)
        if (max_pol - min_pol) < 0.2:
            feedback.append(
                "The emotional arc is relatively flat. Consider adding tonal variation."
            )

    if polarities and polarities[-1] < 0.1:
        feedback.append(
            "The narrative ends on a neutral or low sentiment note. Consider a stronger conclusion."
        )

    return feedback or ["Great job! The narrative flow seems well-structured."]


@app.post("/analyze-narrative")
def analyze_narrative_endpoint(request: TextRequest):
    polarities = analyze_sentiment(request.text)
    feedback = get_structural_feedback(polarities)

    filename = f"plot_{uuid.uuid4()}.png"
    create_sentiment_plot(polarities, filename)

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0

    return {
        "analysis_summary": {
            "num_sentences": len(polarities),
            "average_polarity": round(avg_polarity, 2),
        },
        "feedback": feedback,
        "emotional_arc_plot_url": f"/plots/{filename}",
    }


@app.get("/")
def read_root():
    return {"status": "online", "message": "Narrative Flow AI operational"}