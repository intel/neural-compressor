import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from vllm_qdq_plugin.sage3_attn.impl import Sage3TritonImpl
from vllm_qdq_plugin.sage3_attn.routing import patch_attention as P
from vllm_qdq_plugin.sage3_attn.routing.route_table import RouteTable


def _write_route(data: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, f)
    f.close()
    return f.name


def _make_impl():
    impl = mock.MagicMock(spec=Sage3TritonImpl)
    impl.forward_cuda.return_value = "CUDA"
    impl._forward_sdpa.return_value = "SDPA"
    return impl


def _make_module(role="self", layer_idx=0, impl=None):
    return types.SimpleNamespace(
        role=role,
        layer_idx=layer_idx,
        attention=impl if impl is not None else _make_impl(),
    )


ORIG_SENTINEL = "ORIG"


def _fake_orig(self, q, k, v, m):
    return ORIG_SENTINEL


class RoutedDispatchTests(unittest.TestCase):
    """Exercise _routed_run_local_attention branches with fakes (no GPU)."""

    def setUp(self) -> None:
        # Snapshot module globals so each test runs in isolation.
        self._saved = (P._installed, P._ORIG, P._TABLE)
        P._TABLE = RouteTable.from_json_file(
            _write_route(
                {
                    "default": "mxfp4",
                    "rules": [
                        {"steps": "0-3", "config": "sdpa"},
                        {"layers": "20-39", "config": "mxfp8_s1"},
                    ],
                }
            )
        )
        P._ORIG = _fake_orig
        self.q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)

    def tearDown(self) -> None:
        P._installed, P._ORIG, P._TABLE = self._saved

    def _run(self, module, q=None):
        q = self.q if q is None else q
        return P._routed_run_local_attention(module, q, q, q, None)

    def _with_step(self, step):
        return (
            mock.patch.object(P, "is_forward_context_available", return_value=True),
            mock.patch.object(
                P,
                "get_forward_context",
                return_value=types.SimpleNamespace(denoise_step_idx=step),
            ),
        )

    def test_float32_falls_through_to_original(self) -> None:
        qf = torch.zeros(1, 4, 2, 8, dtype=torch.float32)
        self.assertEqual(self._run(_make_module(), qf), ORIG_SENTINEL)

    def test_non_self_role_falls_through(self) -> None:
        self.assertEqual(self._run(_make_module(role="cross")), ORIG_SENTINEL)

    def test_non_sage_impl_falls_through(self) -> None:
        module = types.SimpleNamespace(role="self", layer_idx=0, attention=object())
        self.assertEqual(self._run(module), ORIG_SENTINEL)

    def test_sdpa_token_calls_forward_sdpa(self) -> None:
        avail, ctx = self._with_step(2)
        impl = _make_impl()
        with avail, ctx:
            result = self._run(_make_module(layer_idx=25, impl=impl))
        self.assertEqual(result, "SDPA")
        impl._forward_sdpa.assert_called_once()
        impl.forward_cuda.assert_not_called()

    def test_layer_rule_passes_config_override(self) -> None:
        avail, ctx = self._with_step(50)  # outside step rule
        impl = _make_impl()
        with avail, ctx:
            result = self._run(_make_module(layer_idx=25, impl=impl))
        self.assertEqual(result, "CUDA")
        _, kwargs = impl.forward_cuda.call_args
        self.assertEqual(kwargs.get("config_override"), "mxfp8_s1")

    def test_default_config_used_when_no_rule_matches(self) -> None:
        avail, ctx = self._with_step(50)
        impl = _make_impl()
        with avail, ctx:
            self._run(_make_module(layer_idx=5, impl=impl))
        _, kwargs = impl.forward_cuda.call_args
        self.assertEqual(kwargs.get("config_override"), "mxfp4")

    def test_no_forward_context_treats_step_as_none(self) -> None:
        impl = _make_impl()
        with mock.patch.object(P, "is_forward_context_available", return_value=False):
            self._run(_make_module(layer_idx=25, impl=impl))
        _, kwargs = impl.forward_cuda.call_args
        self.assertEqual(kwargs.get("config_override"), "mxfp8_s1")

    def test_none_route_falls_through_to_original(self) -> None:
        # No default + no matching rule -> resolve() returns None -> original.
        P._TABLE = RouteTable.from_json_file(_write_route({"rules": [{"layers": "20-39", "config": "mxfp4"}]}))
        avail, ctx = self._with_step(0)
        impl = _make_impl()
        with avail, ctx:
            result = self._run(_make_module(layer_idx=5, impl=impl))
        self.assertEqual(result, ORIG_SENTINEL)
        impl.forward_cuda.assert_not_called()


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        from vllm_omni.diffusion.attention.layer import Attention

        self.Attention = Attention
        self._saved_method = Attention._run_local_attention
        self._saved_globals = (P._installed, P._ORIG, P._TABLE)
        # Force a clean, uninstalled state for the test.
        P._installed = False
        P._ORIG = None
        P._TABLE = None

    def tearDown(self) -> None:
        self.Attention._run_local_attention = self._saved_method
        P._installed, P._ORIG, P._TABLE = self._saved_globals

    def test_install_patches_method_and_is_idempotent(self) -> None:
        path = _write_route({"default": "mxfp4", "rules": [{"steps": "0-3", "config": "sdpa"}]})
        try:
            original = self.Attention._run_local_attention
            P.install(path)
            self.assertIs(
                self.Attention._run_local_attention,
                P._routed_run_local_attention,
            )
            self.assertIs(P._ORIG, original)
            self.assertTrue(P._installed)
            self.assertEqual(P._TABLE.default, "mxfp4")

            # Second install is a no-op: globals unchanged.
            P.install(path)
            self.assertIs(P._ORIG, original)
        finally:
            os.unlink(path)

    def test_install_raises_on_bad_route_file(self) -> None:
        path = _write_route({"rules": [{"config": "not_real"}]})
        try:
            with self.assertRaises(ValueError):
                P.install(path)
            # Failed install must not leave the method patched.
            self.assertFalse(P._installed)
            self.assertIs(self.Attention._run_local_attention, self._saved_method)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
