import unittest

from seer.anomaly_detection.detectors.mp_scorers import MPIRQScorer


class TestMPScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = MPIRQScorer()

    def test_batch_score(self):

        # TODO: Import timeseries with dummy values
        # ts = []
        # mp_dist = []
        # sensitivities = ["low", "medium", "high"]  # unused in method
        # direction = ["up", "down", "both"]  # unused in method

        # window_size = 0
        pass

    def test_stream_score(self):
        pass

        # TODO: Similar to batch_scores but output is a single value for score and flag
