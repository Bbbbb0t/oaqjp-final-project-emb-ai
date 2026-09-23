from flask import Flask, jsonify, render_template, request

from emotion_detection import EmotionDetectionError, emotion_detector

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/emotionDetector")
def detect_emotion():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400

    try:
        result = emotion_detector(text)
    except EmotionDetectionError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
