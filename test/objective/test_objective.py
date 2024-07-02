"""Tests for neural_compressor quantization."""

import unittest


class TestObjs(unittest.TestCase):

    def test_tune_data(self):
        from neural_compressor.objective import MultiObjective

        obj = MultiObjective(
            objectives=["accuracy", "modelsize", "performance"],
            accuracy_criterion={"relative": 0.1},
            obj_criterion=[True, False, False],
            obj_weight=[0.7, 0.2, 0.1],
        )
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]],
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 4)

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"], {"relative": 0.1}, obj_criterion=[True, False, False]
        )
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]],
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"], {"absolute": 0.3}, obj_criterion=[True, False, False]
        )
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]],
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            objectives=["accuracy", "modelsize", "performance"],
            accuracy_criterion={"absolute": 0.3},
            obj_criterion=[True, False, False],
            obj_weight=[0.6, 0.1, 0.3],
        )
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 400, 0.24]],
            [0.750, [0.750, 400, 0.22]],
            [0.720, [0.720, 400, 0.18]],
            [0.790, [0.790, 400, 0.15]],
            [0.750, [0.750, 400, 0.24]],
            [0.785, [0.785, 400, 0.13]],
        ]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"],
            {"absolute": 0.04, "higher_is_better": False},
            obj_weight=[0.6, 0.1, 0.3],
        )
        baseline = [0.75, [0.75, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 400, 0.10]],
            [0.750, [0.750, 400, 0.22]],
            [0.720, [0.720, 400, 0.18]],
            [0.790, [0.790, 400, 0.15]],
            [0.750, [0.750, 400, 0.24]],
            [0.785, [0.785, 400, 0.13]],
        ]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 3)

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"],
            {"absolute": 0.4, "higher_is_better": False},
            obj_weight=[0.6, 0.1, 0.3],
        )
        baseline = [0.0, [0.0, 780, 0.6]]
        tune_data = [
            [0.00, [0.00, 400, 0.23]],
            [0.80, [0.80, 400, 0.10]],
            [0.02, [0.02, 400, 0.22]],
            [0.10, [0.10, 400, 0.18]],
            [0.20, [0.20, 400, 0.15]],
            [0.00, [0.00, 400, 0.24]],
            [0.50, [0.50, 400, 0.13]],
        ]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 0)

        obj = MultiObjective(
            ["modelsize", "performance"], {"relative": 0.08}, obj_criterion=[False], obj_weight=[0.2, 0.8]
        )
        baseline = [0.8, [780, 0.6]]
        tune_data = [
            [0.760, [400, 0.23]],
            [0.778, [420, 0.24]],
            [0.750, [430, 0.22]],
            [0.720, [410, 0.18]],
            [0.790, [360, 0.15]],
            [0.750, [430, 0.24]],
            [0.785, [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

    def test_multi_obj_metric(self):
        from neural_compressor.objective import MultiObjective

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"],
            {"relative": 0.04, "higher_is_better": True},
            metric_criterion=[True, True],
            metric_weight=[0.0, 1.0],
            obj_criterion=[True, False, False],
            obj_weight=[0.6, 0.1, 0.3],
        )
        baseline = [[0.75, 0.4], [[0.75, 0.4], 780, 0.6]]
        tune_data = [
            [[0.760, 0.4], [[0.760, 0.4], 400, 0.23]],
            [[0.778, 0.3], [[0.778, 0.3], 400, 0.10]],
            [[0.750, 0.3], [[0.750, 0.3], 400, 0.22]],
            [[0.720, 0.3], [[0.720, 0.3], 400, 0.18]],
            [[0.790, 0.3], [[0.790, 0.3], 400, 0.15]],
            [[0.750, 0.3], [[0.750, 0.3], 400, 0.24]],
            [[0.785, 0.3], [[0.785, 0.3], 400, 0.13]],
        ]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 0)

        obj = MultiObjective(
            ["accuracy", "modelsize", "performance"],
            {"absolute": 0.4, "higher_is_better": False},
            metric_criterion=[False, True],
            obj_weight=[0.6, 0.1, 0.3],
        )
        baseline = [[0.0, 0.9], [[0.0, 0.9], 780, 0.6]]
        tune_data = [
            [[0.00, 0.9], [[0.00, 0.9], 400, 0.23]],
            [[0.80, 0.8], [[0.80, 0.8], 400, 0.10]],
            [[0.02, 0.7], [[0.02, 0.7], 400, 0.22]],
            [[0.10, 0.6], [[0.10, 0.6], 400, 0.18]],
            [[0.20, 0.7], [[0.20, 0.7], 400, 0.15]],
            [[0.00, 0.7], [[0.00, 0.7], 400, 0.24]],
            [[0.50, 0.7], [[0.50, 0.7], 400, 0.13]],
        ]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 0)

        obj = MultiObjective(
            ["modelsize", "performance"],
            {"relative": 0.08},
            metric_criterion=[True, True],
            metric_weight=[0.5, 0.5],
            obj_weight=[0.2, 0.8],
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            ["modelsize", "performance"],
            {"absolute": 0.013},
            metric_criterion=[True, True],
            metric_weight=[0.5, 0.5],
            obj_weight=[0.2, 0.8],
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 4)

        obj = MultiObjective(
            ["modelsize", "performance"], {"relative": 0.08}, metric_criterion=[True, True], obj_weight=[0.2, 0.8]
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 4)

        obj = MultiObjective(
            ["modelsize", "performance"], {"absolute": 0.06}, metric_criterion=[True, True], obj_weight=[0.2, 0.8]
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            ["modelsize", "performance"], {"relative": 0.08}, metric_criterion=[True, False], obj_weight=[0.2, 0.8]
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(
            ["modelsize", "performance"], {"absolute": 0.07}, metric_criterion=[True, False], obj_weight=[0.2, 0.8]
        )
        baseline = [[0.8, 0.1], [780, 0.6]]
        tune_data = [
            [[0.760, 0.093], [400, 0.23]],
            [[0.778, 0.094], [420, 0.24]],
            [[0.750, 0.092], [430, 0.22]],
            [[0.720, 0.093], [410, 0.18]],
            [[0.790, 0.093], [360, 0.15]],
            [[0.750, 0.093], [430, 0.24]],
            [[0.785, 0.060], [360, 0.13]],
        ]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)


if __name__ == "__main__":
    unittest.main()
