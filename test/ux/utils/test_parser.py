# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Log parsers test."""

import unittest
from unittest.mock import MagicMock, patch

from neural_compressor.ux.utils.parser import BenchmarkParserFactory, OptimizationParser


class TestTuningParser(unittest.TestCase):
    """TuningParser tests."""

    def test_parsing_empty_file_list(self) -> None:
        """Test parsing of none files."""
        optimization_parser = OptimizationParser([])
        parsed = optimization_parser.process()

        self.assertEqual({}, parsed)

    @patch("builtins.open", create=True)
    def test_parsing_empty_files(self, mocked_open: MagicMock) -> None:
        """Test parsing of files without any lines."""
        mocked_open.return_value.__enter__.return_value = []

        optimization_parser = OptimizationParser(["file.log"])
        parsed = optimization_parser.process()

        self.assertEqual({}, parsed)

    @patch("builtins.open", create=True)
    def test_parsing_simple_file(self, mocked_open: MagicMock) -> None:
        """Test parsing of file."""
        mocked_open.return_value.__enter__.return_value = [
            "Foo bar baz",
            "2021-05-27 07:52:50 [INFO] Tune 1 result is: [accuracy: 0.1234, duration (seconds): 5.6789], Best tune result is: None",  # noqa: E501
            "2021-05-27 07:52:50 [INFO] Tune 2 result is: [accuracy: 0.99876, duration (seconds): 0.5432], Best tune result is: [accuracy: 0.99876, duration (seconds): 0.5432]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] FP32 baseline is: [accuracy: 0.12344, duration (seconds): 5.6789]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] Save quantized model to /foo/bar/baz.pb.",
            "a b c d",
        ]

        optimization_parser = OptimizationParser(["file.log"])
        parsed = optimization_parser.process()

        self.assertEqual(
            {
                "acc_input_model": 0.1234,
                "acc_optimized_model": 0.9988,
                "path_optimized_model": "/foo/bar/baz.pb",
            },
            parsed,
        )

    @patch("builtins.open", create=True)
    def test_parsing_file_with_duplicated_lines(self, mocked_open: MagicMock) -> None:
        """Test parsing of files without any lines."""
        mocked_open.return_value.__enter__.return_value = [
            "Foo bar baz",
            "2021-05-27 07:52:50 [INFO] Tune 1 result is: [accuracy: 0.1234, duration (seconds): 5.6789], Best tune result is: None",  # noqa: E501
            "2021-05-27 07:52:50 [INFO] Tune 2 result is: [accuracy: 0.2345, duration (seconds): 0.6789], Best tune result is: [accuracy: 0.2345, duration (seconds): 0.6789]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] FP32 baseline is: [accuracy: 0.12344, duration (seconds): 5.6789]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] Save quantized model to /a/b/c.pb.",
            "a b c d",
            "2021-05-27 07:52:50 [INFO] Tune 1 result is: [accuracy: 0.1234, duration (seconds): 5.6789], Best tune result is: None",  # noqa: E501
            "2021-05-27 07:52:50 [INFO] Tune 2 result is: [accuracy: 0.99876, duration (seconds): 0.5432], Best tune result is: [accuracy: 0.99876, duration (seconds): 0.5432]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] FP32 baseline is: [accuracy: 0.12344, duration (seconds): 5.6789]",  # noqa: E501
            "2021-05-27 07:52:27 [INFO] Save quantized model to /foo/bar/baz.pb.",
        ]

        tuning_parser = OptimizationParser(["file.log"])
        parsed = tuning_parser.process()

        self.assertEqual(
            {
                "acc_input_model": 0.1234,
                "acc_optimized_model": 0.9988,
                "path_optimized_model": "/foo/bar/baz.pb",
            },
            parsed,
        )


class TestBenchmarkParser(unittest.TestCase):
    """BenchmarkParser tests."""

    def test_parsing_empty_file_list(self) -> None:
        """Test parsing of none files."""
        benchmark_parser = BenchmarkParserFactory.get_parser(
            benchmark_mode="performance",
            logs=[],
        )
        parsed = benchmark_parser.process()

        self.assertEqual({}, parsed)

    @patch("builtins.open", create=True)
    def test_parsing_empty_files(self, mocked_open: MagicMock) -> None:
        """Test parsing of files without any lines."""
        mocked_open.return_value.__enter__.return_value = []

        benchmark_parser = BenchmarkParserFactory.get_parser(
            benchmark_mode="performance",
            logs=["file.log"],
        )
        parsed = benchmark_parser.process()

        self.assertEqual({}, parsed)

    @patch("builtins.open", create=True)
    def test_parsing_simple_file(self, mocked_open: MagicMock) -> None:
        """Test parsing of file."""
        mocked_open.return_value.__enter__.return_value = [
            "Foo bar baz",
            "performance mode benchmark result:",
            "2021-04-14 05:16:09 [INFO] Accuracy is 0.1234567",
            "2021-04-14 05:16:09 [INFO] Batch size = 1234",
            "2021-04-14 05:16:09 [INFO] Latency: 2.34567 ms",
            "2021-04-14 05:16:09 [INFO] Throughput: 123.45678 images/sec",
            "2021-04-14 05:16:10 [INFO] performance mode benchmark done!",
            "2021-04-14 05:16:10 [INFO]",
            "performance mode benchmark result:",
            "a b c d",
        ]

        benchmark_parser = BenchmarkParserFactory.get_parser(
            benchmark_mode="performance",
            logs=["file.log"],
        )
        parsed = benchmark_parser.process()

        self.assertEqual(
            {
                "perf_throughput_input_model": 123.4568,
                "perf_throughput_optimized_model": 123.4568,
                "perf_latency_input_model": 2.3457,
                "perf_latency_optimized_model": 2.3457,
            },
            parsed,
        )

    @patch("builtins.open", create=True)
    def test_parsing_simple_file_with_many_entries(self, mocked_open: MagicMock) -> None:
        """Test parsing of file."""
        mocked_open.return_value.__enter__.return_value = [
            "Foo bar baz",
            "performance mode benchmark result:",
            "2021-04-14 05:16:09 [INFO] Accuracy is 0.1234567",
            "2021-04-14 05:16:09 [INFO] Batch size = 1234",
            "2021-04-14 05:16:09 [INFO] Latency: 1.0 ms",
            "2021-04-14 05:16:09 [INFO] Latency: 2.0 ms",
            "2021-04-14 05:16:09 [INFO] Latency: 3.0 ms",
            "2021-04-14 05:16:09 [INFO] Latency: 4.0 ms",
            "2021-04-14 05:16:09 [INFO] Latency: 5.0 ms",
            "2021-04-14 05:16:09 [INFO] Latency: 6.0 ms",
            "2021-04-14 05:16:09 [INFO] Throughput: 1.0 images/sec",
            "2021-04-14 05:16:09 [INFO] Throughput: 2.0 images/sec",
            "2021-04-14 05:16:09 [INFO] Throughput: 3.0 images/sec",
            "2021-04-14 05:16:09 [INFO] Throughput: 4.0 images/sec",
            "2021-04-14 05:16:09 [INFO] Throughput: 5.0 images/sec",
            "2021-04-14 05:16:10 [INFO] performance mode benchmark done!",
            "2021-04-14 05:16:10 [INFO]",
            "performance mode benchmark result:",
            "a b c d",
        ]

        benchmark_parser = BenchmarkParserFactory.get_parser(
            benchmark_mode="performance",
            logs=["file.log"],
        )
        parsed = benchmark_parser.process()

        self.assertEqual(
            {
                "perf_throughput_input_model": 15.0,  # SUM
                "perf_throughput_optimized_model": 15.0,  # SUM
                "perf_latency_input_model": 3.5,  # AVG
                "perf_latency_optimized_model": 3.5,  # AVG
            },
            parsed,
        )


if __name__ == "__main__":
    unittest.main()
