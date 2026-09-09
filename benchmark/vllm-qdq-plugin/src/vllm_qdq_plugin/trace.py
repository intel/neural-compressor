# SPDX-License-Identifier: Apache-2.0
"""Optional trace logging for QDQ calls.

Enable with VLLM_QDQ_TRACE=1.
"""

from vllm.logger import init_logger

from . import envs

logger = init_logger(__name__)

_call_count = 0
_logged_formats: set[str] = set()


def log_qdq_once(format_name: str, dtype, *, use_cute: bool) -> None:
    """Log the selected QDQ backend once per format in each process."""
    if format_name in _logged_formats:
        return
    _logged_formats.add(format_name)
    logger.info("QDQ runtime: format=%s dtype=%s cute=%s", format_name, dtype, use_cute)


def trace_qdq(op_name: str, shape, dtype):
    """Print a trace line if tracing is enabled."""
    if not envs.VLLM_QDQ_TRACE:
        return
    global _call_count
    _call_count += 1
    if _call_count <= 200:
        print(f"[QDQ] op={op_name} shape={shape} dtype={dtype}")
