# Watson Emotion Detection Web Application

A Flask application that analyzes customer feedback with the Watson NLP emotion service. It returns scores for anger, disgust, fear, joy, and sadness, together with the dominant emotion.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The default Watson endpoint is the Skills Network lab endpoint. To use another compatible endpoint, set `WATSON_EMOTION_URL`; to select a different model, set `WATSON_MODEL_ID`.

## API

```bash
curl -X POST http://localhost:5000/emotionDetector \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"I love this product"}'
```

## Test and lint

```bash
python -m unittest discover -v
flake8 app.py emotion_detection.py test_emotion_detection.py
```

The service returns HTTP 400 for missing/blank input, HTTP 502 when Watson cannot be reached or returns malformed data, and HTTP 404/405 for unsupported routes or methods.
