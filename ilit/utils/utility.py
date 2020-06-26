"""quantization auto-tuning config system.

This file specifies default config options for quantization auto-tuning tool.
User should not change values in this file. Instead, user should write a config
file (in yaml) and use cfg_from_file(yaml_file) to load it and override the default
options.
"""

import os
import os.path as osp
import inspect
import time
import yaml
import sys

def print_info():
    print(inspect.stack()[1][1],":",inspect.stack()[1][2],":", inspect.stack()[1][3])

def cfg_from_file(file_name):
    """Load a config file and merge it into the default options.

       Args:
           file_name (string): The name of configuration yaml file
    """
    with open(file_name, 'r') as f:
        yaml_cfg = yaml.load(f, yaml.SafeLoader)

    return yaml_cfg

def caller_obj(obj_name):
    """Get the object of local variable, function or class in caller.

       Args:
           obj_name (string): The name of object in caller
    """
    for f in inspect.stack()[1:]:
        if obj_name in f[0].f_locals:
            return f[0].f_locals[obj_name]

class LazyImport(object):
    """Lazy import python module till use

       Args:
           module_name (string): The name of module imported later
    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(self.module_name)

        return getattr(self.module, name)

class AverageMeter(object):
    """Computes the average value

       Args:
           skip (optional, integar): the skipped iterations in calculation
    """
    def __init__(self, skip=0):
        self.skip = skip
        self.reset()

    def reset(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val):
        self.cnt += 1
        if self.cnt <= self.skip:
            self.avg = 0
            return
        self.sum += val
        self.avg = self.sum / (self.cnt - self.skip)

class Timeout(object):
    """Timeout class to check if spending time is beyond target.

       Args:
           seconds (optional, integar): the timeout value, 0 means early stop.
    """
    def __init__(self, seconds=0):
        self.seconds = seconds

    def __enter__(self):
        self.die_after = time.time() + self.seconds
        return self

    def __exit__(self, type, value, traceback):
        pass

    @property
    def timed_out(self):
        return time.time() > self.die_after

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif isinstance(obj, list):
        size += sum([get_size(item, seen) for item in obj])
    else:
        size += sum([get_size(v, seen) for v in dir(obj)])

    return size
