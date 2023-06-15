import unittest
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from seer.trend_detection.detectors.cusum_detection import CUSUMDetector

class TestCusumDetector(unittest.TestCase):

    def get_sample_data(self):
        trend_data = {"data": [[1681934400, [{"count": 680.0}]], [1681938000, [{"count": 742.0}]],
                               [1681941600, [{"count": 631.0}]], [1681945200, [{"count": 567.0}]],
                               [1681948800, [{"count": 538.0}]], [1681952400, [{"count": 619.0}]],
                               [1681956000, [{"count": 577.0}]], [1681959600, [{"count": 551.5}]],
                               [1681963200, [{"count": 589.0}]], [1681966800, [{"count": 531.5}]],
                               [1681970400, [{"count": 562.0}]], [1681974000, [{"count": 528.0}]],
                               [1681977600, [{"count": 587.0}]], [1681981200, [{"count": 569.0}]],
                               [1681984800, [{"count": 615.0}]], [1681988400, [{"count": 611.0}]],
                               [1681992000, [{"count": 630.0}]], [1681995600, [{"count": 637.0}]],
                               [1681999200, [{"count": 661.0}]], [1682002800, [{"count": 671.0}]],
                               [1682006400, [{"count": 638.0}]], [1682010000, [{"count": 670.0}]],
                               [1682013600, [{"count": 666.5}]], [1682017200, [{"count": 650.0}]],
                               [1682020800, [{"count": 641.0}]], [1682024400, [{"count": 682.0}]],
                               [1682028000, [{"count": 627.0}]], [1682031600, [{"count": 611.5}]],
                               [1682035200, [{"count": 557.0}]], [1682038800, [{"count": 526.0}]],
                               [1682042400, [{"count": 538.0}]], [1682046000, [{"count": 598.0}]],
                               [1682049600, [{"count": 563.0}]], [1682053200, [{"count": 588.0}]],
                               [1682056800, [{"count": 610.5}]], [1682060400, [{"count": 576.0}]],
                               [1682064000, [{"count": 598.0}]], [1682067600, [{"count": 560.5}]],
                               [1682071200, [{"count": 623.0}]], [1682074800, [{"count": 557.0}]],
                               [1682078400, [{"count": 883.5}]], [1682082000, [{"count": 972.0}]],
                               [1682085600, [{"count": 844.5}]], [1682089200, [{"count": 929.0}]],
                               [1682092800, [{"count": 1071.0}]], [1682096400, [{"count": 1090.0}]],
                               [1682100000, [{"count": 883.0}]], [1682103600, [{"count": 913.0}]],
                               [1682107200, [{"count": 850.0}]], [1682110800, [{"count": 911.5}]],
                               [1682114400, [{"count": 814.0}]], [1682118000, [{"count": 786.0}]],
                               [1682121600, [{"count": 660.5}]], [1682125200, [{"count": 605.5}]],
                               [1682128800, [{"count": 551.0}]], [1682132400, [{"count": 430.0}]],
                               [1682136000, [{"count": 635.0}]], [1682139600, [{"count": 569.0}]],
                               [1682143200, [{"count": 491.5}]], [1682146800, [{"count": 536.0}]],
                               [1682150400, [{"count": 533.0}]], [1682154000, [{"count": 393.0}]],
                               [1682157600, [{"count": 534.0}]], [1682161200, [{"count": 498.0}]],
                               [1682164800, [{"count": 645.5}]], [1682168400, [{"count": 521.0}]],
                               [1682172000, [{"count": 485.5}]], [1682175600, [{"count": 668.0}]],
                               [1682179200, [{"count": 654.0}]], [1682182800, [{"count": 520.5}]],
                               [1682186400, [{"count": 619.5}]], [1682190000, [{"count": 549.5}]],
                               [1682193600, [{"count": 560.0}]], [1682197200, [{"count": 550.5}]],
                               [1682200800, [{"count": 604.5}]], [1682204400, [{"count": 623.0}]],
                               [1682208000, [{"count": 561.0}]], [1682211600, [{"count": 598.0}]],
                               [1682215200, [{"count": 743.5}]], [1682218800, [{"count": 658.0}]],
                               [1682222400, [{"count": 704.0}]], [1682226000, [{"count": 606.0}]],
                               [1682229600, [{"count": 508.0}]], [1682233200, [{"count": 486.0}]],
                               [1682236800, [{"count": 554.0}]], [1682240400, [{"count": 543.0}]],
                               [1682244000, [{"count": 435.0}]], [1682247600, [{"count": 561.5}]],
                               [1682251200, [{"count": 518.0}]], [1682254800, [{"count": 661.0}]],
                               [1682258400, [{"count": 514.5}]], [1682262000, [{"count": 581.5}]],
                               [1682265600, [{"count": 503.0}]], [1682269200, [{"count": 598.0}]],
                               [1682272800, [{"count": 520.5}]], [1682276400, [{"count": 494.0}]],
                               [1682280000, [{"count": 785.0}]], [1682283600, [{"count": 383.0}]],
                               [1682287200, [{"count": 457.0}]], [1682290800, [{"count": 464.0}]],
                               [1682294400, [{"count": 559.0}]], [1682298000, [{"count": 489.5}]],
                               [1682301600, [{"count": 746.0}]], [1682305200, [{"count": 609.0}]],
                               [1682308800, [{"count": 587.0}]], [1682312400, [{"count": 1263.5}]],
                               [1682316000, [{"count": 744.5}]], [1682319600, [{"count": 805.5}]],
                               [1682323200, [{"count": 987.0}]], [1682326800, [{"count": 869.0}]],
                               [1682330400, [{"count": 779.5}]], [1682334000, [{"count": 880.5}]],
                               [1682337600, [{"count": 929.5}]], [1682341200, [{"count": 862.0}]],
                               [1682344800, [{"count": 884.0}]], [1682348400, [{"count": 895.0}]],
                               [1682352000, [{"count": 939.0}]], [1682355600, [{"count": 1183.0}]],
                               [1682359200, [{"count": 922.0}]], [1682362800, [{"count": 953.0}]],
                               [1682366400, [{"count": 1373.5}]], [1682370000, [{"count": 963.0}]],
                               [1682373600, [{"count": 719.5}]], [1682377200, [{"count": 1024.5}]],
                               [1682380800, [{"count": 940.0}]], [1682384400, [{"count": 630.0}]],
                               [1682388000, [{"count": 943.0}]], [1682391600, [{"count": 796.5}]],
                               [1682395200, [{"count": 695.5}]], [1682398800, [{"count": 965.5}]],
                               [1682402400, [{"count": 921.5}]], [1682406000, [{"count": 896.0}]],
                               [1682409600, [{"count": 962.0}]], [1682413200, [{"count": 1099.0}]],
                               [1682416800, [{"count": 837.0}]], [1682420400, [{"count": 915.0}]],
                               [1682424000, [{"count": 978.5}]], [1682427600, [{"count": 1051.5}]],
                               [1682431200, [{"count": 1125.0}]], [1682434800, [{"count": 838.5}]],
                               [1682438400, [{"count": 936.0}]], [1682442000, [{"count": 1170.0}]],
                               [1682445600, [{"count": 1057.5}]], [1682449200, [{"count": 1097.0}]],
                               [1682452800, [{"count": 1034.0}]], [1682456400, [{"count": 1219.0}]],
                               [1682460000, [{"count": 936.0}]], [1682463600, [{"count": 911.0}]],
                               [1682467200, [{"count": 841.0}]], [1682470800, [{"count": 790.0}]],
                               [1682474400, [{"count": 1015.0}]], [1682478000, [{"count": 651.5}]],
                               [1682481600, [{"count": 839.0}]], [1682485200, [{"count": 820.0}]],
                               [1682488800, [{"count": 783.0}]], [1682492400, [{"count": 853.0}]],
                               [1682496000, [{"count": 811.0}]], [1682499600, [{"count": 971.0}]],
                               [1682503200, [{"count": 931.0}]], [1682506800, [{"count": 1028.0}]],
                               [1682510400, [{"count": 828.0}]], [1682514000, [{"count": 817.0}]],
                               [1682517600, [{"count": 971.0}]], [1682521200, [{"count": 1235.0}]],
                               [1682524800, [{"count": 1080.0}]], [1682528400, [{"count": 974.0}]],
                               [1682532000, [{"count": 1016.0}]], [1682535600, [{"count": 938.0}]],
                               [1682539200, [{"count": 738.5}]], [1682542800, [{"count": 924.0}]],
                               [1682546400, [{"count": 900.0}]], [1682550000, [{"count": 958.0}]],
                               [1682553600, [{"count": 974.0}]], [1682557200, [{"count": 756.0}]],
                               [1682560800, [{"count": 912.0}]], [1682564400, [{"count": 924.0}]],
                               [1682568000, [{"count": 822.0}]], [1682571600, [{"count": 776.0}]],
                               [1682575200, [{"count": 979.0}]], [1682578800, [{"count": 606.0}]],
                               [1682582400, [{"count": 1109.5}]], [1682586000, [{"count": 884.5}]],
                               [1682589600, [{"count": 833.0}]], [1682593200, [{"count": 897.0}]],
                               [1682596800, [{"count": 844.0}]], [1682600400, [{"count": 1014.0}]]],
                      "start": 1681934400, "end": 1683144000}
        timestamps = [x[0] for x in trend_data["data"]]
        metrics = [x[1][0]['count'] for x in trend_data["data"]]

        input_data = pd.DataFrame(
            {
                'time': timestamps,
                'y': metrics
            }
        )

        return input_data

    def setUp(self):
        self.data = self.get_sample_data()
        self.cusum_detector = CUSUMDetector(self.data, self.data)

    def test_get_changepoint(self):
        change_point = self.cusum_detector._get_change_point(np.asarray(self.data['y']), 10, None, "increase")

        actual_changepoint = 104
        actual_breakpoint = 1682308800

        assert actual_changepoint == change_point.changepoint
        assert actual_breakpoint == change_point.changetime

    def test_get_llr(self):
        llr = self.cusum_detector._get_llr(np.asarray(self.data['y']), 618.9, 925.6, 104)

        actual_value = 153.062
        assert actual_value == round(llr, 3)

    def test_changepoints_returned(self):
        changepoints = self.cusum_detector.detector()

        num_changepoints = 1
        assert num_changepoints == len(changepoints)

    def test_detector(self):
        changepoints = self.cusum_detector.detector()
        changepoint = changepoints[0]

        actual_value = {'start_time': 1682308800, 'end_time': 1682308800, 'confidence': 1.0, 'direction': 'increase',
                        'delta': 306.6691358024691, 'regression_detected': True, 'stable_changepoint': True,
                        'mu0': 618.9666666666667, 'mu1': 925.6358024691358, 'llr': 153.06251960483866,
                        'llr_int': np.inf, 'p_value': 0.0, 'p_value_int': np.nan}

        expected_value = {'start_time': changepoint.start_time, 'end_time': changepoint.end_time,
                          'confidence': changepoint.confidence, 'direction': changepoint.direction,
                          'delta': changepoint.delta, 'regression_detected': changepoint.regression_detected,
                          'stable_changepoint': changepoint.stable_changepoint, 'mu0': changepoint.mu0,
                          'mu1': changepoint.mu1, 'llr': changepoint.llr, 'llr_int': changepoint.llr_int,
                          'p_value': changepoint.p_value, 'p_value_int': changepoint.p_value_int}

        assert actual_value == expected_value
