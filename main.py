import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline

emotion_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events cleanly without blocking API initialization.
    """
    global emotion_classifier

    print("Loading Hugging Face emotion classifier model...")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1
    )
    print("Model loaded successfully!")
    yield
    print("Shutting down application...")

app = FastAPI(
    title="Narrative Flow AI",
    description="High-performance async engine for deep emotional arc and narrative analysis.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Text to analyse. Must contain at least 10 characters."
    )

class SentenceEmotionPoint(BaseModel):
    sentence_index: int
    text: str
    dominant_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]

class NarrativeMetrics(BaseModel):
    total_sentences: int
    primary_arc_emotion: str
    emotional_variance: float
    pacing_intensity: float

class AnalysisResponse(BaseModel):
    metrics: NarrativeMetrics
    sentence_analysis: List[SentenceEmotionPoint]
    structural_feedback: List[str]

def split_text_into_sentences(text: str) -> List[str]:
    """
    Helper function to split input text into clean sentences.
    Handles standard punctuation and strips extraneous whitespace.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s', text.strip())
    return [s.strip() for s in sentences if s.strip()]

async def classify_sentence_async(sentence: str) -> List[Dict[str, Any]]:
    """
    Wraps the synchronous ML pipeline call inside an asyncio event loop thread pool.
    Prevents heavy CPU inference from blocking other HTTP requests.
    """
    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(None, emotion_classifier, sentence)

    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], list):
            return raw_output[0]
        return raw_output
    return []

def calculate_narrative_metrics(sentence_data: List[SentenceEmotionPoint]) -> NarrativeMetrics:
    """
    Computes mathematical metrics accross the emotional arc using NumPy.
    """
    if not sentence_data:
        return NarrativeMetrics(
            total_sentences=0,
            primary_arc_emotion="neutral",
            emotional_variance=0.0,
            pacing_intensity=0.0
        )

    confidences = [item.confidence for item in sentence_data]

    variance = float(np.var(confidences)) if len(confidences) > 1 else 0.0

    intensity = float(np.mean(confidences))

    emotion_counts = {}
    for item in sentence_data:
        emotion_counts[item.dominant_emotion] = emotion_counts.get(item.dominant_emotion, 0) + 1

    primary_emotion = max(emotion_counts, key=emotion_counts.get)

    return NarrativeMetrics(
        total_sentences=len(sentence_data),
        primary_arc_emotion=primary_emotion,
        emotional_variance=round(variance, 4),
        pacing_intensity=round(intensity, 4)
    )

def generate_structual_feedback(metrics: NarrativeMetrics, sentence_data: List[SentenceEmotionPoint]) -> List[str]:
    """
    Generates dynamic heuristics and narrative suggestions based on metrics.
    """
    feedback = []

    if metrics.total_sentences < 3:
        feedback.append("Text is short. Add more sentences to create a full emotional trajectory.")
        return feedback

    if metrics.emotional_variance < 0.01:
        feedback.append("Monotonic Arc detected: THe story maintains alomost constant emotional tone. Introduce conflicts or shifts to enhance engagement.")

    final_emotion = sentence_data[-1].dominant_emotion
    if final_emotion in ["sadness", "fear", "anger"]:
        feedback.append(f"Tense Ending: Your narrative resolves with '{final_emotion}'. Ensure this aligns with your desired payoff.")
    elif final_emotion == "joy":
        feedback.append("Uplifiting Ending: The story concludes on a high emotional note.")


    if metrics.pacing_intensity > 0.85:
        feedback.append("High Emotional Intensity: Sentences express strongly pronounced feelings throughout.")

    return feedback or ["Well-balanced narrative progression!"]


@app.post("/analyze-narrative", response_model=AnalysisResponse)
async def analyze_narrative(request: AnalysisRequest):
    """
    Asynchronous REST endpoint that accepts text and outputs dynamic emotion trajectory JSON.
    """
    sentences = split_text_into_sentences(request.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Unable to extract meaningfulsentences from text.")

    tasks = [classify_sentence_async(sentence) for sentence in sentences]
    raw_results = await asyncio.gather(*tasks)

    sentence_analysis: List[SentenceEmotionPoint] = []

    for idx, (sentence, scores) in enumerate(zip(sentences, raw_results)):
        if isinstance(scores, list):
            emotion_map = {
                item["label"]: round(float(item["score"]), 4) 
                for item in scores 
                if isinstance(item, dict) and "label" in item and "score" in item
            }
        else:
            emotion_map = {}

        if not emotion_map:
            continue

        dominant_label = max(emotion_map, key=emotion_map.get)
        dominant_confidence = emotion_map[dominant_label]

        sentence_analysis.append(
            SentenceEmotionPoint(
                sentence_index=idx,
                text=sentence,
                dominant_emotion=dominant_label,
                confidence=dominant_confidence,
                emotion_scores=emotion_map
            )
        )

    metrics = calculate_narrative_metrics(sentence_analysis)
    feedback = generate_structual_feedback(metrics, sentence_analysis)

    return AnalysisResponse(
        metrics=metrics,
        sentence_analysis=sentence_analysis,
        structural_feedback=feedback
    )

@app.get("/")
async def health_check():
    """
    Health check endpoint for production load balancers.
    """
    return {"status": "healthy", "model_loaded": emotion_classifier is not None}

