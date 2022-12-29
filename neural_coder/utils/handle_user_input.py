# Copyright (c) 2022 Intel Corporation
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
from typing import List
from .. import globals
import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


def get_all_code_path(user_input: str) -> List:
    user_code_path = get_user_code_path(user_input)
    if globals.consider_imports:
        import_path = get_imports_path(user_code_path)
    else:
        import_path = []

    # if import_path intersects user_code_path, clear import_path: this is
    # for cases where there is import of self folder and we only care about 
    # the main file (user_code_path) itself
    if len(list(set(user_code_path).intersection(set(import_path)))) > 0:
        import_path = []

    all_code_path = user_code_path + import_path
    logger.debug(f"Number of code files to analyze: {len(all_code_path)}")

    num_user_code_path = len(user_code_path)

    return all_code_path, num_user_code_path


def get_user_code_path(user_input: str) -> List:
    list_path = []

    # detect whether (a list of files)/file/folder/url
    global user_input_type
    if type(user_input) == list:
        user_input_type = "a list of files"
    elif "github.com" in user_input and ".py" in user_input:
        user_input_type = "url_py"
    elif ".py" in user_input:
        user_input_type = "file"
    elif "github.com" in user_input:
        user_input_type = "url_repo"
    else:
        user_input_type = "folder"

    logger.debug(f"user input code type: {user_input_type}")

    # get list of file path
    if user_input_type == "url_repo":
        from git import Repo
        Repo.clone_from(user_input,  "./cloned_github_repo")
        dir_input = "./cloned_github_repo"
    if user_input_type == "folder":
        dir_input = user_input

    if user_input_type == "file":
        list_path.append(os.path.abspath(user_input))
    elif user_input_type == "url_py":
        import requests
        user_input = user_input.replace("github.com", "raw.githubusercontent.com").replace("/blob","")
        r = requests.get(user_input)
        save_py_path = "./neural_coder_workspace/model.py"
        f = open(save_py_path, "wb")
        f.write(r.content)
        list_path.append(os.path.abspath(save_py_path))
    elif user_input_type == "a list of files":
        list_path += [os.path.abspath(i) for i in user_input]
    else:
        for path, dir_list, file_list in os.walk(dir_input):
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                if file_path[-3:] == ".py" and file_path[-11:] != "__init__.py" and file_path[-8:] != "setup.py":
                    list_path.append(os.path.abspath(file_path))

    return list_path


def get_imports_path(user_code_path: List) -> List:

    pip_name_exceptions = [
        "argparse",
        "ast",
        "classy_vision",
        "collections",
        "contextlib",
        "data",
        "dataclasses",
        "dataloader",
        "dataset",
        "datasets",
        "einops",
        "enum",
        "evaluate",
        "fx",
        "glob",
        "h5py",
        "imageio",
        "importlib",
        "inspect",
        "intel_extension_for_pytorch",
        "itertools",
        "jax",
        "json",
        "keras",
        "libcst",
        "linecache",
        "logging",
        "loggings",
        "math",
        "model",
        "models",
        "neural_coder",
        "neural_compressor",
        "numpy",
        "os",
        "onnxruntime",
        "pandas",
        "PIL",
        "random",
        "requests",
        "shutil",
        "subprocess",
        "sys",
        "system",
        "tempfile",
        "typing",
        "tensorflow",
        "threading",
        "time",
        "timeit",
        "torch",
        "torchdynamo",
        "tqdm",
        "traceback",
        "trainer_qa",
        "unittest",
        "utils_qa",
        "vissl",
        "warning",
        "warnings",
        "packaging",
        "pytorch_lightning",
        "__future__",
    ]

    if globals.cache_load_transformers:
        pip_name_exceptions.append("transformers")

    list_pip_path = []
    list_pip_name = []

    # get list of pip name
    for path in user_code_path:
        lines = open(path, 'r').read().split('\n')
        for line in lines:
            is_import_line = False
            if line[0:6] == "import" and line[0:8] != "import ." and "," not in line:  # to-do: handle "," case
                is_import_line = True
                start = 7
            elif line[0:4] == "from" and line[0:6] != "from .":
                is_import_line = True
                start = 5
            if is_import_line:
                space_idx = line[start:].find(" ")
                dot_idx = line[start:].find(".")
                if space_idx == -1 and dot_idx == -1:
                    pip_name = line[start:]
                elif space_idx > 0 and dot_idx == -1:
                    pip_name = line[start: start + space_idx]
                elif space_idx == -1 and dot_idx > 0:
                    pip_name = line[start: start + dot_idx]
                elif space_idx > 0 and dot_idx > 0:
                    pip_name = line[start: start + min(space_idx, dot_idx)]
                list_pip_name.append(pip_name)
    list_pip_name = list(set(list_pip_name).difference(set(pip_name_exceptions)))
    for item in list_pip_name:
        if "_nc" in item:
            list_pip_name.remove(item)

    logger.debug(f"list pip name: {list_pip_name}")

    # get list of pip path
    cmd_import = " ".join(["import " + i + ";" for i in list_pip_name])
    try:
        exec(cmd_import)
    except ModuleNotFoundError as mnfe:
        logger.error(
            f"Please install all required pip modules defined in your Python scripts "
            f"before running Neural Coder: {mnfe}"
        )
        quit()

    import inspect
    for i in list_pip_name:
        try:
            pip_dir_path = inspect.getsourcefile(eval(i))
            pip_dir_path = pip_dir_path[0:pip_dir_path.rfind("/")]
            for path, dir_list, file_list in os.walk(pip_dir_path):
                for file_name in file_list:
                    file_path = os.path.join(path, file_name)
                    if file_path[-3:] == ".py" and file_path[-11:] != "__init__.py" and file_path[-8:] != "setup.py":
                        list_pip_path.append(os.path.abspath(file_path))
        except TypeError as te:
            logger.error(
                f"Please reinstall certain pip modules as its detected as a built-in module "
                f"and the installation path cannot be retrieved: {te}"
            )

    return list_pip_path
