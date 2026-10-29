# app/schemas/narrative.py

from typing import Dict, List, Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Text to analyze. Must contain at least 10 characters."
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