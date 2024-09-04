import unittest

from seer.anomaly_detection.detectors.mp_utils import MPUtils


class TestMPUtils(unittest.TestCase):

    def setUp(self):
        self.scorer = MPUtils()

    # TODO: Fix the mocks
    # @patch(MPConfig)
    # @patch(normalizer)
    def test_get_mp_dist_from_mp(self):
        # mp = []
        # pad_to_len = []  # TODO: determine if should loop over multiple vals
        pass
        # self.scorer.get_mp_dist_from_mp(mp, pad_to_len, pad_to_len, mp_config, normalizer)

        # TODO: Test cases
        # mp_config is none
        # mp_config is not none and normalize_np is not none
        # normalizer is none --> exception
        # pad_to_len is none
        # all values are fine
