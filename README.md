# NarratorFlowAI 📖📈

**An AI-powered narrative analysis platform that explores emotional trajectories, pacing intensity, and structural patterns in written text.**

NarratorFlowAI is an asynchronous NLP microservice that analyzes the emotional evolution of narratives at the sentence level. It combines transformer-based emotion classification, numerical analytics, and rule-based structural heuristics to help users understand how emotions develop throughout a story.

🔗 **Live Demo:** [NarratorFlowAI](https://medha170.github.io/NarratorFlowAI/)

⚙️ **Backend API:** [NarratorFlowAI API](https://narratorflowai.onrender.com)

📚 **API Documentation:** [Swagger UI](https://narratorflowai.onrender.com/docs)

---

## 🚀 Features

- **Sentence-Level Emotion Classification:** Context-aware emotion detection across seven distinct emotional dimensions using a fine-tuned Hugging Face transformer model.

- **Optimized ML Inference:** Uses a 79 MB INT8-quantized ONNX model with ONNX Runtime and CPU execution, reducing inference overhead and deployment footprint.

- **Asynchronous ML Concurrency:** Non-blocking inference execution using Python's `asyncio`, thread pool executors (`run_in_executor`), and `asyncio.gather` to prevent computationally intensive inference from blocking the event loop.

- **Advanced Narrative Analytics:** Computes statistical metrics using NumPy:
  - Total sentence count
  - Primary arc emotion
  - Emotional variance (measuring tonal spread and dramatic shifts)
  - Pacing intensity (measuring overall emotional magnitude)

- **Automated Structural Heuristics:** Generates rule-based narrative feedback covering emotional balance, climax resolution, and pacing.

- **Resilient Caching Layer:** SHA-256 text hashing backed by Redis to avoid redundant inference, with a fail-safe fallback when Redis is unavailable.

- **Interactive Visualization Dashboard:** A browser-based frontend featuring multi-line emotional trajectory visualizations powered by Chart.js.

---

## 🛠️ Tech Stack

| Component | Technologies |
|---|---|
| Backend Framework | FastAPI, Uvicorn |
| API Validation | Pydantic v2 |
| Machine Learning | Hugging Face Transformers, ONNX Runtime |
| Model Optimization | INT8 Quantization |
| NLP Model | `j-hartmann/emotion-english-distilroberta-base` |
| Numerical Processing | NumPy |
| Asynchronous Processing | asyncio, ThreadPoolExecutor |
| Caching | Redis, hashlib (SHA-256) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data Visualization | Chart.js |
| Deployment | Render (Backend), GitHub Pages (Frontend) |

---

## 🏗️ Architecture

NarratorFlowAI follows a modular service-oriented architecture.

1. **Input Processing:** The frontend sends narrative text to the FastAPI backend.
2. **Text Tokenization:** The text service splits the narrative into individual sentences.
3. **Emotion Inference:** The model service performs sentence-level emotion classification using the optimized ONNX model.
4. **Asynchronous Execution:** Inference tasks are executed concurrently without blocking the FastAPI event loop.
5. **Caching:** Previously analyzed inputs can be retrieved using SHA-256-based cache keys.
6. **Metric Computation:** NumPy calculates emotional variance, pacing intensity, and other narrative metrics.
7. **Structural Analysis:** Rule-based heuristics generate narrative feedback.
8. **Visualization:** The frontend renders the resulting emotional trajectories and analysis.

---

## 📂 Project Structure

```text
NarrativeFlowAI/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── narrative.py    # API route handlers
│   │
│   ├── core/
│   │   └── config.py               # Global configuration
│   │
│   ├── schemas/
│   │   └── narrative.py            # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── cache_service.py        # Redis caching & SHA-256 hashing
│   │   ├── metrics_service.py      # NumPy metrics & heuristics
│   │   ├── model_service.py        # ML lifecycle & inference pipeline
│   │   └── text_service.py         # Sentence tokenization
│   │
│   └── main.py                     # FastAPI application entry point
│
├── tests/
│   └── test_main.py
│
├── index.html                      # Interactive Chart.js dashboard
├── requirements.txt
├── mcp.yaml
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Medha170/NarrativeFlowAI.git
cd NarrativeFlowAI
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Redis Setup (Optional)

Redis is used for caching repeated narrative requests.

To run Redis locally using Docker:

```bash
docker run -d -p 6379:6379 redis
```

**Note:** Redis is optional. The application includes a fail-safe fallback and continues operating when Redis is unavailable.

---

## ▶️ Running the Backend Locally

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **Base URL:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs

The model is loaded lazily when inference is first required.

### Connecting the Frontend Locally

The frontend's `API_URL` constant in `index.html` currently points to the deployed Render backend.

To test against your local backend, temporarily change it to:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

Restore the deployed URL when using the hosted frontend.

---

## 📡 API Reference

### `POST /analyze-narrative`

Accepts narrative text and returns sentence-level emotion predictions, aggregate metrics, and structural feedback.

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

*The example illustrates the response structure; actual predictions and scores depend on the input narrative.*

---

## 🌐 Deployment

NarratorFlowAI uses separate hosting platforms for its frontend and backend.

| Component | Platform | URL |
|---|---|---|
| Frontend | GitHub Pages | [Live Demo](https://medha170.github.io/NarrativeFlowAI/) |
| Backend | Render | [API Service](https://narratorflowai.onrender.com) |
| API Documentation | FastAPI / Swagger | [Interactive Docs](https://narratorflowai.onrender.com/docs) |

The GitHub Pages frontend communicates with the deployed FastAPI backend through HTTP requests.

---

## 🧪 Testing

Run the backend test suite using:

```bash
python -m unittest discover -s tests -v
```

---

## 💡 Future Enhancements

- Expand narrative analysis with deeper structural and thematic insights.
- Introduce comparative analysis between multiple narrative drafts.
- Explore additional model optimization techniques for faster inference.
- Extend visualization capabilities to support richer narrative exploration.

---

## 👩‍💻 Author

**Medha Shree**

Computer Science Undergraduate | AI/ML & Software Engineering Enthusiast

[GitHub](https://github.com/Medha170)

---

*Built to explore the emotional architecture of stories, one sentence at a time.* 📖