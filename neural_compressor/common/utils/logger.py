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


import functools
import logging
import os

from neural_compressor.common.utils import Mode

__all__ = [
    "level",
    "level_name",
    "Logger",  # TODO: not expose it
    "logger",
    "TuningLogger",
]


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

    @staticmethod
    def log(level, msg, *args, **kwargs):
        """Output log with the level as a parameter."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().log(level, line, *args, **kwargs)
        else:
            Logger().get_logger().log(level, msg, *args, **kwargs)

    @staticmethod
    def debug(msg, *args, **kwargs):
        """Output log with the debug level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().debug(line, *args, **kwargs)
        else:
            Logger().get_logger().debug(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """Output log with the error level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().error(line, *args, **kwargs)
        else:
            Logger().get_logger().error(msg, *args, **kwargs)

    @staticmethod
    def fatal(msg, *args, **kwargs):
        """Output log with the fatal level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().fatal(line, *args, **kwargs)
        else:
            Logger().get_logger().fatal(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        """Output log with the info level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().info(line, *args, **kwargs)
        else:
            Logger().get_logger().info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        """Output log with the warning level (Alias of the method warn)."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().warning(line, *args, **kwargs)
        else:
            Logger().get_logger().warning(msg, *args, **kwargs)

    @functools.lru_cache(None)
    def warning_once(msg, *args, **kwargs):
        """Output log with the warning level only once."""
        Logger.warning("Below warning will be shown only once:")
        Logger.warning(msg, *args, **kwargs)


level = Logger().get_logger().level
level_name = logging.getLevelName(level)

logger = Logger


def _get_log_msg(mode):
    log_msg = None
    if mode == Mode.QUANTIZE:
        log_msg = "Quantization"
    elif mode == Mode.PREPARE:  # pragma: no cover
        log_msg = "Preparation"
    elif mode == Mode.CONVERT:  # pragma: no cover
        log_msg = "Conversion"
    elif mode == Mode.LOAD:  # pragma: no cover
        log_msg = "Loading"
    return log_msg


class TuningLogger:
    """A unified logger for the tuning/quantization process.

    It assists validation teams in retrieving logs.
    """

    @classmethod
    def tuning_start(cls) -> None:
        """Log the start of the tuning process."""
        logger.info("Tuning started.")

    @classmethod
    def trial_start(cls, trial_index: int = None) -> None:
        """Log the start of a trial."""
        logger.info("%d-trail started.", trial_index)

    @classmethod
    def execution_start(cls, mode=Mode.QUANTIZE, stacklevel=2):
        """Log the start of the execution process."""
        log_msg = _get_log_msg(mode)
        assert log_msg is not None, "Please check `mode` in execution_start function of TuningLogger class."
        logger.info("{} started.".format(log_msg), stacklevel=stacklevel)

    @classmethod
    def execution_end(cls, mode=Mode.QUANTIZE, stacklevel=2):
        """Log the end of the execution process."""
        log_msg = _get_log_msg(mode)
        assert log_msg is not None, "Please check `mode` in execution_end function of TuningLogger class."
        logger.info("{} end.".format(log_msg), stacklevel=stacklevel)

    @classmethod
    def evaluation_start(cls) -> None:
        """Log the start of the evaluation process."""
        logger.info("Evaluation started.")

    @classmethod
    def evaluation_end(cls) -> None:
        """Log the end of the evaluation process."""
        logger.info("Evaluation end.")

    @classmethod
    def trial_end(cls, trial_index: int = None) -> None:
        """Log the end of a trial."""
        logger.info("%d-trail end.", trial_index)

    @classmethod
    def tuning_end(cls) -> None:
        """Log the end of the tuning process."""
        logger.info("Tuning completed.")
