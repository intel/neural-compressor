import os
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
    "eval(",
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

def _wrap_subprocess(func, timeout):
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
        p.join(timeout)
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

def secure_eval_func(user_func, timeout=300):
    """
    Return a secured version of user eval_func.
    Controlled by env NC_EVAL_SANDBOX (default=1 to enable).
    """
    enable = os.getenv("NC_EVAL_SANDBOX", "1") == "1"
    if not isinstance(user_func, FunctionType):
        logger.warning("Provided eval_func is not a plain function; security checks limited.")
        return user_func
    try:
        _static_check(user_func)
    except ValueError as e:
        raise
    if not enable:
        logger.info("Sandbox disabled (NC_EVAL_SANDBOX!=1); eval_func will run directly.")
        return user_func
    # Check picklability for subprocess
    try:
        mp.get_context("spawn")  # ensure spawn context available
        import pickle
        pickle.dumps(user_func)
    except Exception:
        logger.warning("eval_func not picklable; cannot sandbox. Running directly (re-enable by refactoring).")
        return user_func
    return _wrap_subprocess(user_func, timeout)
