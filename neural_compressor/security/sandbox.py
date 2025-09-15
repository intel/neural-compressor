# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import multiprocessing as mp
from types import FunctionType

from ..utils import logger

_FORBIDDEN_PATTERNS = [
    "import os",
    "import subprocess",
    "import sys",
    "subprocess.",
    "os.system",
    "os.popen",
    "popen(",
    "Popen(",
    "system(",
    "exec(",
    "__import__(",
]


def _static_check(func):
    try:
        src = inspect.getsource(func)
    except (OSError, IOError):  # pragma: no cover
        logger.warning("Cannot read source of eval_func; skip static scan.")
        return
    lowered = src.lower()
    for p in _FORBIDDEN_PATTERNS:
        if p in lowered:
            raise ValueError(f"Unsafe token detected in eval_func: {p}")


def _wrap_subprocess(func):
    def wrapper(model, *args, **kwargs):
        q = mp.Queue()

        def _target():
            try:
                res = func(model, *args, **kwargs)
                q.put(("ok", res))
            except Exception as e:  # pragma: no cover
                q.put(("err", repr(e)))

        p = mp.Process(target=_target)
        p.start()
        p.join()
        if p.is_alive():
            p.terminate()
            raise TimeoutError("eval_func execution timeout.")
        if q.empty():
            raise RuntimeError("eval_func produced no result.")
        status, val = q.get()
        if status == "err":
            raise RuntimeError(f"eval_func raised exception: {val}")
        return val

    wrapper.__name__ = getattr(func, "__name__", "secure_eval_wrapper")
    return wrapper


def secure_eval_func(user_func):
    """Return a secured version of user eval_func."""
    if not isinstance(user_func, FunctionType):
        logger.warning("Provided eval_func is not a plain function; security checks limited.")
        return user_func
    try:
        _static_check(user_func)
    except ValueError as e:
        raise
    # Check picklability for subprocess
    try:
        mp.get_context("spawn")  # ensure spawn context available
        import pickle

        pickle.dumps(user_func)
    except Exception:
        logger.warning("eval_func not picklable; cannot sandbox. Running directly (re-enable by refactoring).")
        return user_func
    return _wrap_subprocess(user_func)
