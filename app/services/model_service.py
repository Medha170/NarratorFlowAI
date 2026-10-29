# app/services/model_service.py

import asyncio
from typing import List, Dict, Any
from transformers import pipeline
from core.config import MODEL_NAME

emotion_classifier = None


def load_emotion_model():
    """Initializes and loads the Hugging Face transformer model globally."""
    global emotion_classifier
    print("Loading Hugging Face emotion classifier model...")
    emotion_classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        return_all_scores=True,
        device=-1
    )
    print("Model loaded successfully!")


async def classify_sentence_async(sentence: str) -> List[Dict[str, Any]]:
    """Offloads CPU-bound transformer pipeline inference to a background thread pool."""
    if not emotion_classifier:
        raise RuntimeError("Model is not initialized.")
        
    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(None, emotion_classifier, sentence)
    
    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], list):
            return raw_output[0]
        return raw_output
    return []