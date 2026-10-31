# app/api/v1/endpoints/narrative.py

import asyncio
from typing import List
from fastapi import APIRouter, HTTPException
from schemas.narrative import AnalysisRequest, AnalysisResponse, SentenceEmotionPoint
from services.text_service import split_text_into_sentences
from services.model_service import classify_sentence_async, emotion_classifier
from services.metrics_service import calculate_narrative_metrics, generate_structural_feedback
from services.cache_service import generate_text_hash, get_cached_analysis, cache_analysis

router = APIRouter()


@router.post("/analyze-narrative", response_model=AnalysisResponse)
async def analyze_narrative(request: AnalysisRequest):
    """Processes narrative text through model, text, and metrics services asynchronously."""
    
    text_hash = generate_text_hash(request.text)
    cached_response = get_cached_analysis(text_hash)
    
    if cached_response:
        print("Cache HIT: Returning stored analysis instantly.")
        return cached_response

    print("Cache MISS: Running full NLP pipeline inference...")

    sentences = split_text_into_sentences(request.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Unable to extract meaningful sentences from text.")

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
    feedback = generate_structural_feedback(metrics, sentence_analysis)

    response_data = AnalysisResponse(
        metrics=metrics,
        sentence_analysis=sentence_analysis,
        structural_feedback=feedback
    )

    cache_analysis(text_hash, response_data)

    return response_data