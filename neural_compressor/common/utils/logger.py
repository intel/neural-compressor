#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""Logger: handles logging functionalities."""

import logging
import os

__all__ = [
    "level",
    "log",
    "info",
    "debug",
    "warning",
    "error",
    "fatal",
    "Logger",  # TODO: not expose it
    "logger",
]


class Logger(object):
    """Logger class."""

    __instance = None

    def __new__(cls):
        """Create a singleton Logger instance."""
        if Logger.__instance is None:
            Logger.__instance = object.__new__(cls)
            Logger.__instance._log()
        return Logger.__instance

    def _log(self):
        """Setup the logger format and handler."""
        LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
        self._logger = logging.getLogger("neural_compressor")
        self._logger.handlers.clear()
        self._logger.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)
        self._logger.propagate = False

    def get_logger(self):
        """Get the logger."""
        return self._logger


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


level = Logger().get_logger().level

logger = Logger().get_logger()


def log(level, msg, *args, **kwargs):
    """Output log with the level as a parameter."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().log(level, line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().log(level, msg, *args, **kwargs, stacklevel=2)


def debug(msg, *args, **kwargs):
    """Output log with the debug level."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().debug(line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().debug(msg, *args, **kwargs, stacklevel=2)


def error(msg, *args, **kwargs):
    """Output log with the error level."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().error(line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().error(msg, *args, **kwargs, stacklevel=2)


def fatal(msg, *args, **kwargs):
    """Output log with the fatal level."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().fatal(line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().fatal(msg, *args, **kwargs, stacklevel=2)


def info(msg, *args, **kwargs):
    """Output log with the info level."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().info(line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().info(msg, *args, **kwargs, stacklevel=2)


def warning(msg, *args, **kwargs):
    """Output log with the warning level (Alias of the method warn)."""
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split("\n")):
            Logger().get_logger().warning(line, *args, **kwargs, stacklevel=2)
    else:
        Logger().get_logger().warning(msg, *args, **kwargs, stacklevel=2)
