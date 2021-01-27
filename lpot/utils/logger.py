#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import os
import logging


class Logger(object):
    __instance = None

    def __new__(cls):
        if Logger.__instance is None:
            Logger.__instance = object.__new__(cls)
            Logger.__instance._log()
        return Logger.__instance

    def _log(self):
        LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
        self._logger = logging.getLogger()
        self._logger.handlers.clear()
        self._logger.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            "%Y-%m-%d %H:%M:%S")
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
