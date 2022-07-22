# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convenience function for logging compliance tags to stdout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import logging
import json
import os
import re
import sys
import time
import uuid

from mlperf_compliance.tags import *

ROOT_DIR_GNMT = None
ROOT_DIR_MASKRCNN = None
ROOT_DIR_MINIGO = None
ROOT_DIR_NCF = None

# Set by imagenet_main.py
ROOT_DIR_RESNET = None

ROOT_DIR_SSD = None

# Set by transformer_main.py and process_data.py
ROOT_DIR_TRANSFORMER = None


PATTERN = re.compile('[a-zA-Z0-9]+')

LOG_FILE = os.getenv("COMPLIANCE_FILE")
# create logger with 'spam_application'
LOGGER = logging.getLogger('mlperf_compliance')
LOGGER.setLevel(logging.DEBUG)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
_STREAM_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_STREAM_HANDLER)

if LOG_FILE:
  _FILE_HANDLER = logging.FileHandler(LOG_FILE)
  _FILE_HANDLER.setLevel(logging.DEBUG)
  LOGGER.addHandler(_FILE_HANDLER)
else:
  _STREAM_HANDLER.setLevel(logging.DEBUG)



def get_caller(stack_index=2, root_dir=None):
  ''' Returns file.py:lineno of your caller. A stack_index of 2 will provide
      the caller of the function calling this function. Notice that stack_index
      of 2 or more will fail if called from global scope. '''
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub("^" + root_dir + "/", "", filename)
  return "%s:%d" % (filename, caller.lineno)


def _mlperf_print(key, value=None, benchmark=None, stack_offset=0,
                  tag_set=None, deferred=False, root_dir=None,
                  extra_print=False, prefix=""):
  ''' Prints out an MLPerf Log Line.

  key: The MLPerf log key such as 'CLOCK' or 'QUALITY'. See the list of log keys in the spec.
  value: The value which contains no newlines.
  benchmark: The short code for the benchmark being run, see the MLPerf log spec.
  stack_offset: Increase the value to go deeper into the stack to find the callsite. For example, if this
                is being called by a wraper/helper you may want to set stack_offset=1 to use the callsite
                of the wraper/helper itself.
  tag_set: The set of tags in which key must belong.
  deferred: The value is not presently known. In that case, a unique ID will
            be assigned as the value of this call and will be returned. The
            caller can then include said unique ID when the value is known
            later.
  root_dir: Directory prefix which will be trimmed when reporting calling file
            for compliance logging.
  extra_print: Print a blank line before logging to clear any text in the line.
  prefix: String with which to prefix the log message. Useful for
          differentiating raw lines if stitching will be required.

  Example output:
    :::MLP-1537375353 MINGO[17] (eval.py:42) QUALITY: 43.7
  '''

  return_value = None

  if (tag_set is None and not PATTERN.match(key)) or key not in tag_set:
    raise ValueError('Invalid key for MLPerf print: ' + str(key))

  if value is not None and deferred:
    raise ValueError("deferred is set to True, but a value was provided")

  if deferred:
    return_value = str(uuid.uuid4())
    value = "DEFERRED: {}".format(return_value)

  if value is None:
    tag = key
  else:
    str_json = json.dumps(value)
    tag = '{key}: {value}'.format(key=key, value=str_json)

  callsite = get_caller(2 + stack_offset, root_dir=root_dir)
  now = time.time()

  message = '{prefix}:::MLPv0.5.0 {benchmark} {secs:.9f} ({callsite}) {tag}'.format(
      prefix=prefix, secs=now, benchmark=benchmark, callsite=callsite, tag=tag)

  if extra_print:
    print() # There could be prior text on a line

  if tag in STDOUT_TAG_SET:
    LOGGER.info(message)
  else:
    LOGGER.debug(message)

  return return_value


GNMT_TAG_SET = set(GNMT_TAGS)
def gnmt_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=GNMT,
                       stack_offset=stack_offset, tag_set=GNMT_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_GNMT)


MASKRCNN_TAG_SET = set(MASKRCNN_TAGS)
def maskrcnn_print(key, value=None, stack_offset=1, deferred=False,
    extra_print=True, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=MASKRCNN,
                       stack_offset=stack_offset, tag_set=MASKRCNN_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_MASKRCNN, prefix=prefix)


MINIGO_TAG_SET = set(MINIGO_TAGS)
def minigo_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=MINIGO,
                       stack_offset=stack_offset, tag_set=MINIGO_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_MINIGO,
                       prefix=prefix)


NCF_TAG_SET = set(NCF_TAGS)
def ncf_print(key, value=None, stack_offset=1, deferred=False,
              extra_print=True, prefix=""):
  # Extra print is needed for the reference NCF because of tqdm.
  return _mlperf_print(key=key, value=value, benchmark=NCF,
                       stack_offset=stack_offset, tag_set=NCF_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_NCF, prefix=prefix)


RESNET_TAG_SET = set(RESNET_TAGS)
def resnet_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=RESNET,
                       stack_offset=stack_offset, tag_set=RESNET_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_RESNET,
                       prefix=prefix)


SSD_TAG_SET = set(SSD_TAGS)
def ssd_print(key, value=None, stack_offset=1, deferred=False,
              extra_print=True, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=SSD,
                       stack_offset=stack_offset, tag_set=SSD_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_SSD, prefix=prefix)


TRANSFORMER_TAG_SET = set(TRANSFORMER_TAGS)
def transformer_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  return _mlperf_print(key=key, value=value, benchmark=TRANSFORMER,
                       stack_offset=stack_offset, tag_set=TRANSFORMER_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_TRANSFORMER,
                       prefix=prefix)


if __name__ == '__main__':
  ncf_print(EVAL_ACCURACY, {'epoch': 7, 'accuracy': 43.7})
  ncf_print(INPUT_SIZE, 1024)
