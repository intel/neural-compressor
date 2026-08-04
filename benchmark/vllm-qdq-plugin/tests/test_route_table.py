import json
import os
import tempfile
import unittest

from vllm_qdq_plugin.sage3_attn.routing import route_table as rt
from vllm_qdq_plugin.sage3_attn.routing.route_table import RouteTable, _parse_ranges


def _write_route(data: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, f)
    f.close()
    return f.name


class ParseRangesTests(unittest.TestCase):
    def test_mixed_ranges_and_singletons(self) -> None:
        self.assertEqual(
            _parse_ranges("0-7,9,12-15"),
            set(range(0, 8)) | {9} | set(range(12, 16)),
        )

    def test_single_value(self) -> None:
        self.assertEqual(_parse_ranges("5"), {5})

    def test_inclusive_upper_bound(self) -> None:
        self.assertEqual(_parse_ranges("3-3"), {3})

    def test_empty_string_is_empty_set(self) -> None:
        self.assertEqual(_parse_ranges(""), set())

    def test_whitespace_tolerated(self) -> None:
        self.assertEqual(_parse_ranges(" 1 - 3 , 5 "), {1, 2, 3, 5})

    def test_descending_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_ranges("7-3")

    def test_non_integer_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_ranges("a-b")


class RouteTableResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _write_route(
            {
                "default": "mxfp4",
                "rules": [
                    {
                        "layers": "20-39",
                        "steps": "0-7",
                        "config": "mixed_fp8qk_fp4pv",
                    },
                    {"steps": "0-7", "config": "sdpa"},
                    {"layers": "20-39", "config": "mxfp8_s1"},
                ],
            }
        )
        self.table = RouteTable.from_json_file(self.path)

    def tearDown(self) -> None:
        os.unlink(self.path)

    def test_first_rule_wins_when_both_predicates_match(self) -> None:
        self.assertEqual(self.table.resolve(25, 3), "mixed_fp8qk_fp4pv")

    def test_step_only_rule(self) -> None:
        self.assertEqual(self.table.resolve(5, 3), "sdpa")

    def test_layer_only_rule(self) -> None:
        self.assertEqual(self.table.resolve(25, 50), "mxfp8_s1")

    def test_falls_back_to_default(self) -> None:
        self.assertEqual(self.table.resolve(5, 50), "mxfp4")

    def test_none_layer_still_matches_step_rule(self) -> None:
        self.assertEqual(self.table.resolve(None, 3), "sdpa")

    def test_none_step_still_matches_layer_rule(self) -> None:
        self.assertEqual(self.table.resolve(25, None), "mxfp8_s1")

    def test_none_layer_does_not_match_layer_predicate(self) -> None:
        # layer=None can't satisfy rule0's layers predicate; step matches rule1.
        self.assertEqual(self.table.resolve(None, 5), "sdpa")

    def test_both_none_falls_back_to_default(self) -> None:
        self.assertEqual(self.table.resolve(None, None), "mxfp4")


class RouteTableLoadTests(unittest.TestCase):
    def test_no_default_returns_none_on_miss(self) -> None:
        path = _write_route({"rules": [{"layers": "0-5", "config": "mxfp4"}]})
        try:
            table = RouteTable.from_json_file(path)
            self.assertIsNone(table.resolve(99, 0))
            self.assertEqual(table.resolve(3, 0), "mxfp4")
        finally:
            os.unlink(path)

    def test_invalid_config_raises_at_load(self) -> None:
        path = _write_route({"rules": [{"config": "not_a_real_config"}]})
        try:
            with self.assertRaises(ValueError):
                RouteTable.from_json_file(path)
        finally:
            os.unlink(path)

    def test_invalid_default_raises_at_load(self) -> None:
        path = _write_route({"default": "nope", "rules": []})
        try:
            with self.assertRaises(ValueError):
                RouteTable.from_json_file(path)
        finally:
            os.unlink(path)

    def test_missing_config_key_raises(self) -> None:
        path = _write_route({"rules": [{"layers": "0-5"}]})
        try:
            with self.assertRaises(ValueError):
                RouteTable.from_json_file(path)
        finally:
            os.unlink(path)

    def test_sdpa_token_is_allowed(self) -> None:
        path = _write_route({"default": "sdpa", "rules": []})
        try:
            table = RouteTable.from_json_file(path)
            self.assertEqual(table.resolve(0, 0), "sdpa")
        finally:
            os.unlink(path)

    def test_integer_range_spec_is_coerced(self) -> None:
        # JSON numbers for layers/steps should be accepted as single values.
        path = _write_route({"rules": [{"layers": 7, "steps": 0, "config": "mxfp4"}]})
        try:
            table = RouteTable.from_json_file(path)
            self.assertEqual(table.resolve(7, 0), "mxfp4")
            self.assertIsNone(table.resolve(8, 0))
        finally:
            os.unlink(path)

    def test_top_level_must_be_object(self) -> None:
        path = _write_route([])  # a JSON list, not an object
        try:
            with self.assertRaises(ValueError):
                RouteTable.from_json_file(path)
        finally:
            os.unlink(path)

    def test_allowed_configs_includes_sdpa(self) -> None:
        allowed = rt._allowed_configs()
        self.assertIn("sdpa", allowed)
        self.assertIn("mxfp4", allowed)


if __name__ == "__main__":
    unittest.main()
