import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Unit tests for the Watson emotion detector."""

    def _mock_response(self, scores):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{"emotion": scores}]
        }
        return response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_sentence(self, post):
        post.return_value = self._mock_response({
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.01,
            "joy": 0.95,
            "sadness": 0.02,
        })
        result = emotion_detector("I am delighted with this wonderful product.")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_sentence(self, post):
        post.return_value = self._mock_response({
            "anger": 0.92,
            "disgust": 0.02,
            "fear": 0.01,
            "joy": 0.01,
            "sadness": 0.04,
        })
        result = emotion_detector("I am furious about this terrible service.")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_sentence(self, post):
        post.return_value = self._mock_response({
            "anger": 0.02,
            "disgust": 0.91,
            "fear": 0.01,
            "joy": 0.01,
            "sadness": 0.05,
        })
        result = emotion_detector("This product is disgusting and unacceptable.")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_sentence(self, post):
        post.return_value = self._mock_response({
            "anger": 0.02,
            "disgust": 0.01,
            "fear": 0.04,
            "joy": 0.01,
            "sadness": 0.92,
        })
        result = emotion_detector("I am deeply saddened by this disappointing experience.")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_sentence(self, post):
        post.return_value = self._mock_response({
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.93,
            "joy": 0.01,
            "sadness": 0.04,
        })
        result = emotion_detector("I am afraid this product may be unsafe.")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_empty_input_returns_empty_result(self, post):
        result = emotion_detector("")
        self.assertEqual(result, {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        })
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
