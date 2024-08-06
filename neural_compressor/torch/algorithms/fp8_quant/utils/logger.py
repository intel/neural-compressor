# Copyright (c) 2024 Intel Corporation
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

# Taken and adjusted from neural-compressor-fork/neural_compressor/common/utils/logger.py
# Should be merged with INC logger once HQT code is inserted into INC
# TODO: SW-185347 merge INC logger with HQT logger
"""Logger: handles logging functionalities."""


import logging
import os
from logging.handlers import RotatingFileHandler

__all__ = ["logger"]

# Define color escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
WHITE = "\033[37m"
BG_RED = "\033[41m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"


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


logging.TRACE = 5  # There is no 'trace' level for python logger.


def trace(self, msg, *args, **kwargs):
    """Log 'msg % args' with severity 'TRACE'.

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.trace("Houston, we have a %s", "thorny problem", exc_info=1)
    """
    if self.isEnabledFor(logging.TRACE):
        self._log(logging.TRACE, msg, args, **kwargs)


logging.Logger.trace = trace
logging.IGNORE = 60
logging.addLevelName(logging.TRACE, "TRACE")
logging.__all__ += ["TRACE", "trace"]

log_levels = {
    "0": logging.TRACE,  # = 5
    "1": logging.DEBUG,  # = 10
    "2": logging.INFO,  # = 20
    "3": logging.WARNING,  # = 30
    "4": logging.ERROR,  # = 40
    "5": logging.CRITICAL,  # = 50
    "6": logging.IGNORE,  # = 60 (Disabling logger)
}
MAX_LOG_LEVEL_NAME_LEN = 8

DEFAULT_LOG_FILE_SIZE = 1024 * 1024 * 10
DEFAULT_LOG_FILE_AMOUNT = 5


class _Logger(object):
    """_Logger class."""

    __instance = None

    def __new__(cls):
        """Create a singleton _Logger instance."""
        if _Logger.__instance is None:
            _Logger.__instance = object.__new__(cls)
            _Logger.__instance._init_log()
        return _Logger.__instance

    def get_enable_console_val(self):
        enableConsole = os.environ.get("ENABLE_CONSOLE", "False").upper()
        if enableConsole not in ["TRUE", "FALSE", "1", "0"]:
            raise Exception("Env var 'ENABLE_CONSOLE' has to be 'true' or 'false' ('1' or '0' respectively).")
        return enableConsole == "TRUE" or enableConsole == "1"

    def get_log_level(self):
        log_level_str = os.environ.get("LOG_LEVEL_INC", os.environ.get("LOG_LEVEL_ALL"))
        if log_level_str is None:
            return logging.INFO
        if log_level_str not in log_levels:
            raise Exception(f"Wrong Log Level value: '{log_level_str}'. Must be an integer 0-6.")
        return log_levels[log_level_str]

    def prepare_logger_format(self):
        # Time printing is added to format according to the value of PRINT_TIME env var.
        print_time = os.environ.get("PRINT_TIME", "True")
        time_format = "" if print_time.upper() in ["0", "FALSE"] else "%(asctime)s.%(msecs)06d"
        return f"[{time_format}][%(name)s][%(levelname)s] %(message)s"

    # Create a formatter with lower case level name
    @staticmethod
    class LowercaseLevelNameFormatter(logging.Formatter):
        def format(self, record):
            level_name = record.levelname
            record.levelname = record.levelname.lower().ljust(MAX_LOG_LEVEL_NAME_LEN)
            message = super().format(record)
            record.levelname = level_name
            return message

    # Create a formatter with color for the console output
    @staticmethod
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            message = super().format(record)
            # if record.levelname == 'TRACE':
            #   stays black
            if record.levelname == "DEBUG":
                style = CYAN
            elif record.levelname == "INFO":
                style = GREEN
            elif record.levelname == "WARNING":
                style = f"{BOLD}{YELLOW}"
            elif record.levelname == "ERROR":
                style = f"{BOLD}{RED}"
            elif record.levelname == "CRITICAL":
                style = f"{BG_RED}{BOLD}{WHITE}"
            else:
                return message
            return message.replace(
                record.levelname,
                f"{style}{record.levelname.lower().ljust(MAX_LOG_LEVEL_NAME_LEN)}{RESET}",
                1,
            )

    def _init_log(self):
        """Setup the logger format and handler."""
        enableConsole = self.get_enable_console_val()
        self._logger = logging.getLogger("INC")
        log_level = self.get_log_level()
        if log_level == logging.IGNORE:
            self._logger.disabled = True
        else:
            # according to: swtools_sdk/hl_logger/src/hllog_core.cpp
            self._logger.handlers.clear()
            self._logger.setLevel(log_level)
            logging_format = self.prepare_logger_format()
            hls_id = int(os.getenv("HLS_ID", "-1"))
            local_rank_id = int(os.getenv("ID", os.getenv("OMPI_COMM_WORLD_RANK", "-1")))
            habana_logs_path = os.getenv("HABANA_LOGS")
            if habana_logs_path is None:
                habana_logs_path = (
                    "/tmp/.habana_logs" if os.getenv("HOME") is None else os.getenv("HOME") + "/.habana_logs"
                )
            log_folder = f"{habana_logs_path}{''if hls_id < 0 else '/{}'.format(hls_id)}"
            log_folder = f"{log_folder}{''if local_rank_id < 0 else '/{}'.format(local_rank_id)}"
            try:
                os.makedirs(log_folder, exist_ok=True)
            except OSError as error:
                print(
                    f"Warning: Directory '{log_folder}' can not be created for INC logs: {error.strerror}. Logger is disabled."
                )
                self._logger.disabled = True
                pass
            file_path = log_folder + "/inc_log.txt"
            log_file_size = int(os.getenv("INC_LOG_FILE_SIZE", DEFAULT_LOG_FILE_SIZE))
            if log_file_size < 0:
                print(
                    f"Warning: Log file size value is not valid [{log_file_size}]. Using default value [{DEFAULT_LOG_FILE_SIZE}]"
                )
                log_file_size = DEFAULT_LOG_FILE_SIZE
            log_file_amount = int(os.getenv("INC_LOG_FILE_AMOUNT", DEFAULT_LOG_FILE_AMOUNT))
            if log_file_amount < 0:
                print(
                    f"Warning: Log file amount value is not valid [{log_file_amount}]. Using default value [{DEFAULT_LOG_FILE_AMOUNT}]"
                )
                log_file_amount = DEFAULT_LOG_FILE_AMOUNT
            fileHandler = RotatingFileHandler(
                file_path, backupCount=log_file_amount, maxBytes=log_file_size
            )  # default mode = append ("a")
            formatter = _Logger.LowercaseLevelNameFormatter(logging_format, "%Y-%m-%d %H:%M:%S")
            fileHandler.setFormatter(formatter)
            self._logger.addHandler(fileHandler)
            if enableConsole:
                import sys

                streamHandler = logging.StreamHandler(sys.stdout)
                if sys.stdout.isatty():
                    streamHandler.setFormatter(_Logger.ColoredFormatter(logging_format, "%Y-%m-%d %H:%M:%S"))
                else:
                    streamHandler.setFormatter(formatter)
                self._logger.addHandler(streamHandler)
            self._logger.propagate = False

    def log(self, func, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 3)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                func(line, *args, **kwargs)
        else:
            func(msg, *args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        """Output log with the trace level."""
        self.log(self._logger.trace, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Output log with the debug level."""
        self.log(self._logger.debug, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Output log with the info level."""
        self.log(self._logger.info, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Output log with the warning level (Alias of the method warn)."""
        self.log(self._logger.warning, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Output log with the error level."""
        self.log(self._logger.error, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Output log with the critical level."""
        self.log(self._logger.critical, msg, *args, **kwargs)

    fatal = critical


logger = _Logger()
