import unittest
from unittest.mock import Mock, patch

from app import app
from emotion_detection import EmotionDetectionError, emotion_detector


class EmotionDetectionTests(unittest.TestCase):
    @patch("emotion_detection.requests.post")
    def test_detector_returns_scores_and_dominant_emotion(self, post):
        response = Mock()
        response.json.return_value = {
            "emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.02, "fear": 0.03,
                "joy": 0.90, "sadness": 0.04,
            }}]
        }
        post.return_value = response

        result = emotion_detector("I really enjoy this product")

        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.90)
        response.raise_for_status.assert_called_once()

    def test_empty_input_returns_empty_result(self):
        self.assertIsNone(emotion_detector("")["dominant_emotion"])

    @patch("emotion_detection.requests.post")
    def test_invalid_service_response_raises_domain_error(self, post):
        response = Mock()
        response.json.return_value = {}
        post.return_value = response
        with self.assertRaises(EmotionDetectionError):
            emotion_detector("hello")


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_empty_feedback_is_rejected(self):
        response = self.client.post("/emotionDetector", json={"text": " "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    @patch("app.emotion_detector")
    def test_detector_endpoint(self, detector):
        detector.return_value = {"anger": 0, "disgust": 0, "fear": 0, "joy": 1, "sadness": 0, "dominant_emotion": "joy"}
        response = self.client.post("/emotionDetector", json={"text": "great"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["dominant_emotion"], "joy")


if __name__ == "__main__":
    unittest.main()
