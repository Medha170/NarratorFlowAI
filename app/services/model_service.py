# app/services/model_service.py

import asyncio
from typing import List, Dict, Any
from transformers import pipeline
from app.core.config import MODEL_NAME

emotion_classifier = None
_model_lock = asyncio.Lock()


async def load_emotion_model():
    """Initializes and loads the Hugging Face transformer model globally."""
    global emotion_classifier
    if emotion_classifier is None:
        async with _model_lock:
            if emotion_classifier is None:
                print("Lazy-loading Hugging Face model on first request...")

                loop = asyncio.get_running_loop()
                emotion_classifier = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        "text-classification",
                        model=MODEL_NAME,
                        return_all_scores=True,
                        device=-1
                    )
                )
                print("Model lazy-loaded successfully into RAM!")
    return emotion_classifier


async def classify_sentence_async(sentence: str) -> List[Dict[str, Any]]:
    """Ensures the model is loaded, then runs sentence inference asynchronously."""
    classifier = await load_emotion_model()

    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(None, classifier, sentence)

    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], list):
            return raw_output[0]
        return raw_output
    return []