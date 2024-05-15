# Copyright (c) 2023 Intel Corporation
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

import logging
import os


def _pretty_dict(value, indent=0):
    """Make the logger dict pretty."""
    prefix = "\n" + " " * (indent + 4)
    if isinstance(value, dict):
        items = [prefix + repr(key) + ": " + _pretty_dict(value[key], indent + 4) for key in value]
        return "{%s}" % (",".join(items) + "\n" + " " * indent)
    elif isinstance(value, list):
        items = [prefix + _pretty_dict(item, indent + 4) for item in value]
        return "[%s]" % (",".join(items) + "\n" + " " * indent)
    elif isinstance(value, tuple):
        items = [prefix + _pretty_dict(item, indent + 4) for item in value]
        return "(%s)" % (",".join(items) + "\n" + " " * indent)
    else:
        return repr(value)


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
_logger = logging.getLogger("onnx_neural_compressor")
_logger.handlers.clear()
_logger.setLevel(LOGLEVEL)
formatter = logging.Formatter("%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
_logger.addHandler(streamHandler)
_logger.propagate = False


def log(level, msg, *args, **kwargs):
    """Output log with the level as a parameter."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.log(level, line, *args, **kwargs)
    else:
        _logger.log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Output log with the debug level."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.debug(line, *args, **kwargs)
    else:
        _logger.debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Output log with the error level."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.error(line, *args, **kwargs)
    else:
        _logger.error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    """Output log with the fatal level."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.fatal(line, *args, **kwargs)
    else:
        _logger.fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Output log with the info level."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.info(line, *args, **kwargs)
    else:
        _logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Output log with the warning level (Alias of the method warn)."""
    kwargs.setdefault("stacklevel", 2)
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            _logger.warning(line, *args, **kwargs)
    else:
        _logger.warning(msg, *args, **kwargs)
