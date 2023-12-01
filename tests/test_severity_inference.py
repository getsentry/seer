import unittest

from seer.severity.severity_inference import SeverityInference


class TestSeverityInference(unittest.TestCase):
    severity_inference = SeverityInference(
        "models/issue_severity_v0/embeddings", "models/issue_severity_v0/classifier"
    )

    def test_high_severity_error(self):
        score = self.severity_inference.severity_score(
            {"message": "TypeError: bad operand type for unary -: 'str'"}
        )

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.5)

    def test_low_severity_error(self):
        score = self.severity_inference.severity_score(
            {"message": "log: user enjoyed their experience"}
        )

        self.assertIsInstance(score, float)
        self.assertLess(score, 0.5)

    def test_empty_input(self):
        score = self.severity_inference.severity_score({"message": ""})

        self.assertIsInstance(score, float)

    def test_output_range(self):
        test_messages = ["Test message 1", "Another test message", "Yet another test message"]

        for message in test_messages:
            score = self.severity_inference.severity_score({"message": message})

            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_get_embeddings(self):
        embeddings = self.severity_inference.get_embeddings("log: user enjoyed their experience")
        self.assertEqual(len(embeddings), 384)

        embeddings = self.severity_inference.get_embeddings("short")
        self.assertEqual(len(embeddings), 384)

        embeddings = self.severity_inference.get_embeddings(
            "very long gibberish, but how is this going i think it will work right???"
        )
        self.assertEqual(len(embeddings), 384)
