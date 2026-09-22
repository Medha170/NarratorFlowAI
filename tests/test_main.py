import unittest

from main import analyze_sentiment


class SentimentAnalysisTests(unittest.TestCase):
    def test_analyze_sentiment_returns_numeric_polarities(self):
        results = analyze_sentiment("I love this story. It is dark and sad.")

        self.assertGreater(len(results), 0)
        self.assertTrue(all(isinstance(value, float) for value in results))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in results))


if __name__ == "__main__":
    unittest.main()
