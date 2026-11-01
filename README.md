# NarratorFlowAI 📖📈

NarratorFlowAI is a production-grade, asynchronous NLP micro-service that evaluates multi-dimensional emotional arcs, pacing intensity, and narrative structure across sentence timelines.

The project combines a modular FastAPI backend, Hugging Face transformer inference, Redis caching, NumPy analytics, and an interactive Chart.js frontend dashboard.

---

## Features

* **Sentence-Level Emotion Classification:** Context-aware multi-class emotion detection across 7 distinct emotional dimensions using Hugging Face transformers.
* **Asynchronous ML Concurrency:** Non-blocking event loop execution using thread pool executors (`run_in_executor` and `asyncio.gather`) to prevent heavy AI inference from locking the server.
* **Advanced Narrative Analytics:** Computes statistical metrics using NumPy:
* Total sentence count & primary arc emotion
* Emotional variance (measuring tonal spread and dramatic shifts)
* Pacing intensity (measuring overall emotional magnitude)


* **Automated Structural Heuristics:** Generates rule-based narrative feedback suggestions for emotional balance, climax resolution, and pacing.
* **Resilient Caching Layer:** SHA-256 text hashing backed by Redis with a fail-safe fallback to prevent recomputing repeated inputs.
* **Interactive Frontend Demo:** Real-time multi-line trajectory visualization built with Chart.js.

---

## Tech Stack

* **Backend Framework:** FastAPI (with lifespan context manager & CORS middleware)
* **Data Validation:** Pydantic v2
* **Machine Learning:** Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`), PyTorch
* **Numerical Processing:** NumPy
* **Caching:** Redis, hashlib (SHA-256)
* **Frontend:** HTML5, Vanilla JavaScript, Chart.js

---

## Project Structure

```text
NarrativeFlowAI/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── narrative.py    # API route handlers
│   ├── core/
│   │   └── config.py               # Global constants & model configuration
│   ├── schemas/
│   │   └── narrative.py            # Pydantic v2 request/response models
│   ├── services/
│   │   ├── cache_service.py        # Redis caching & SHA-256 text hashing
│   │   ├── metrics_service.py      # NumPy variance, intensity, & heuristics
│   │   ├── model_service.py        # ML lifecycle & async pipeline execution
│   │   └── text_service.py         # Regex sentence tokenization
│   └── main.py                     # FastAPI app, lifespan, & router registration
│
├── tests/
│   └── test_main.py
├── index.html                      # Interactive Chart.js dashboard
├── requirements.txt
├── mcp.yaml
└── README.md

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Medha170/NarrativeFlowAI.git
cd NarrativeFlowAI

```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Running Redis (Optional for Caching)

Ensure Redis is running locally on port `6379` (if using Docker):

```bash
docker run -d -p 6379:6379 redis

```

*(Note: The app features fail-safe fallback and will run normally even if Redis is inactive).*

---

## Running the API

Start the FastAPI server with hot-reload enabled:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

The API will be available at:

* **Base URL:** `[http://127.0.0.1:8000](http://127.0.0.1:8000)`
* **Interactive Docs (Swagger UI):** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

---

## API Reference

### `POST /analyze-narrative`

Analyzes narrative text and returns sentence-level emotion trajectories, summary metrics, and feedback.

**Request Payload:**

```json
{
  "text": "I was heartbroken at first, but the sunrise changed everything. Hope returned and the ending felt warm and hopeful."
}

```

**Response Payload:**

```json
{
  "metrics": {
    "total_sentences": 3,
    "primary_arc_emotion": "joy",
    "emotional_variance": 0.1245,
    "pacing_intensity": 0.8234
  },
  "sentence_analysis": [
    {
      "sentence_index": 0,
      "text": "I was heartbroken at first, but the sunrise changed everything.",
      "dominant_emotion": "sadness",
      "confidence": 0.8612,
      "emotion_scores": {
        "sadness": 0.8612,
        "joy": 0.0743,
        "fear": 0.0211,
        "neutral": 0.1034
      }
    }
  ],
  "structural_feedback": [
    "Uplifting Ending: The story concludes on a high emotional note."
  ]
}

```

---

## Demo UI

A browser-based demo is included in `index.html`. It sends asynchronous requests to the local FastAPI backend and renders a multi-line emotional arc graph.

1. Start the backend: `uvicorn app.main:app --reload`
2. Open `index.html` directly in any web browser.
3. Paste your narrative text and click **Analyze Narrative**.

---