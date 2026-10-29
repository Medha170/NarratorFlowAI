# app/services/metrics_service.py

import numpy as np
from typing import List
from schemas.narrative import SentenceEmotionPoint, NarrativeMetrics


def calculate_narrative_metrics(sentence_data: List[SentenceEmotionPoint]) -> NarrativeMetrics:
    """Computes numerical emotional variance and pacing intensity across sentences using NumPy."""
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


def generate_structural_feedback(metrics: NarrativeMetrics, sentence_data: List[SentenceEmotionPoint]) -> List[str]:
    """Generates structural guidance heuristics based on calculated narrative metrics."""
    feedback = []

    if metrics.total_sentences < 3:
        feedback.append("Text is short. Add more sentences to create a full emotional trajectory.")
        return feedback

    if metrics.emotional_variance < 0.01:
        feedback.append("Monotonic Arc detected: The story maintains an almost constant emotional tone. Introduce conflicts or shifts to enhance engagement.")

    final_emotion = sentence_data[-1].dominant_emotion
    if final_emotion in ["sadness", "fear", "anger"]:
        feedback.append(f"Tense Ending: Your narrative resolves with '{final_emotion}'. Ensure this aligns with your desired payoff.")
    elif final_emotion == "joy":
        feedback.append("Uplifting Ending: The story concludes on a high emotional note.")

    if metrics.pacing_intensity > 0.85:
        feedback.append("High Emotional Intensity: Sentences express strongly pronounced feelings throughout.")

    return feedback or ["Well-balanced narrative progression!"]