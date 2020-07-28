import os
import logging

class Logger(object):
    __instance = None
    def __new__(cls):
        if Logger.__instance is None:
            Logger.__instance = object.__new__(cls)
            Logger.__instance._logger()
        return Logger.__instance

    def _logger(self):
        LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
        self._logger = logging.getLogger()
        self._logger.handlers.clear()
        self._logger.setLevel(LOGLEVEL)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)
        self._logger.propagate = False

    def get_logger(self):
        return self._logger

def _pretty_dict(value, indent=0):
    prefix = '\n' + ' ' * (indent + 4)
    if isinstance(value, dict):
        items = [
            prefix + repr(key) + ': ' + _pretty_dict(value[key], indent + 4)
            for key in value
        ]
        return '{%s}' % (','.join(items) + '\n' + ' ' * indent)
    elif isinstance(value, list):
        items = [
            prefix + _pretty_dict(item, indent + 4)
            for item in value
        ]
        return '[%s]' % (','.join(items) + '\n' + ' ' * indent)
    elif isinstance(value, tuple):
        items = [
            prefix + _pretty_dict(item, indent + 4)
            for item in value
        ]
        return '(%s)' % (','.join(items) + '\n' + ' ' * indent)
    else:
        return repr(value)

level = Logger().get_logger().level
DEBUG = logging.DEBUG

def log(level, msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().log(level, line, *args, **kwargs)
    else:
        Logger().get_logger().log(level, msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().debug(line, *args, **kwargs)
    else:
        Logger().get_logger().debug(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().error(line, *args, **kwargs)
    else:
        Logger().get_logger().error(msg, *args, **kwargs)

def fatal(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().fatal(line, *args, **kwargs)
    else:
        Logger().get_logger().fatal(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().info(line, *args, **kwargs)
    else:
        Logger().get_logger().info(msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().warning(line, *args, **kwargs)
    else:
        Logger().get_logger().warning(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    if isinstance(msg, dict):
        for _, line in enumerate(_pretty_dict(msg).split('\n')):
            Logger().get_logger().warning(line, *args, **kwargs)
    else:
        Logger().get_logger().warning(msg, *args, **kwargs)

