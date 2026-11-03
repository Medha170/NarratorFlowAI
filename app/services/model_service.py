
import asyncio
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from app.core.config import MODEL_NAME

MODEL_DIR = "artifacts/narratorflow-onnx"
MODEL_PATH = f"{MODEL_DIR}/model.onnx"

tokenizer = None
session = None
_model_lock = asyncio.Lock()


async def load_emotion_model():
    global tokenizer, session

    if session is None:
        async with _model_lock:
            if session is None:
                print("Loading quantized ONNX emotion model...")

                tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

                session = ort.InferenceSession(
                    MODEL_PATH,
                    providers=["CPUExecutionProvider"]
                )

                print("ONNX model loaded successfully!")

    return tokenizer, session


async def classify_sentence_async(sentence: str):
    current_tokenizer, current_session = await load_emotion_model()

    inputs = current_tokenizer(
        sentence,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    loop = asyncio.get_running_loop()

    def run_inference():
        outputs = current_session.run(None, dict(inputs))
        logits = outputs[0][0]

        probabilities = np.exp(logits - np.max(logits))
        probabilities = probabilities / probabilities.sum()

        return [
            {
                "label": current_session_label(idx),
                "score": float(score)
            }
            for idx, score in enumerate(probabilities)
        ]

    return await loop.run_in_executor(None, run_inference)


def current_session_label(index):
    from app.core.config import MODEL_NAME
    from transformers import AutoConfig

    if not hasattr(current_session_label, "labels"):
        config = AutoConfig.from_pretrained(MODEL_DIR)
        current_session_label.labels = config.id2label

    return current_session_label.labels[index]
