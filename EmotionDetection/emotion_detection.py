import os
from typing import Any, Dict

import requests


WATSON_EMOTION_URL = os.getenv(
    "WATSON_EMOTION_URL",
    "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
)
WATSON_MODEL_ID = os.getenv(
    "WATSON_MODEL_ID", "emotion_aggregated-workflow_lang_en_stock"
)


def _empty_result() -> Dict[str, Any]:
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


class EmotionDetectionError(RuntimeError):
    """Raised when the Watson emotion service cannot analyze the input."""


def emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    """Return Watson emotion scores and the dominant emotion."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result()

    headers = {"grpc-metadata-mm-model-id": WATSON_MODEL_ID}
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json=payload,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 400:
            return _empty_result()

        response.raise_for_status()
        result = response.json()
        emotions = result["emotionPredictions"][0]["emotion"]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise EmotionDetectionError("Unable to analyze the supplied text") from exc

    scores = {
        emotion: emotions.get(emotion)
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    }
    if any(score is None for score in scores.values()):
        raise EmotionDetectionError("Watson returned an incomplete emotion response")

    return {**scores, "dominant_emotion": max(scores, key=scores.get)}
