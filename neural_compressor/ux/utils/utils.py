# -*- coding: utf-8 -*-
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
"""UX server utils module."""
import csv
import json
import os
import re
import socket
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from werkzeug.utils import secure_filename

from neural_compressor.experimental.metric.metric import registry_metrics
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.utils.consts import WORKDIR_LOCATION, Domains, Frameworks
from neural_compressor.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    InternalException,
    NotFoundException,
)
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.proc import Proc
from neural_compressor.ux.utils.workload.dataloader import Transform
from neural_compressor.version import __version__ as nc_version

dataset_locations = {
    Frameworks.TF: {
        Domains.IMAGE_RECOGNITION: {
            "name": "Imagenet",
            "path": "examples/test/dataset/imagenet",
        },
    },
}

support_boundary_nodes = ["tensorflow"]

configs_directory = os.path.join(
    os.path.dirname(__file__),
    "configs",
)


def deprecated(func: Callable) -> Any:
    """Signal deprecated function."""

    @wraps(func)
    def report_deprecated(*args: str, **kwargs: str) -> Any:
        log.warning(f"Call to deprecated function {func.__name__}.")
        return func(*args, **kwargs)

    return report_deprecated


def is_hidden(path: str) -> bool:
    """Check if path is for hidden filesystem entry."""
    return "." == os.path.basename(path)[0]


@deprecated
def get_dataset_path(framework: str, domain: str) -> str:
    """Get dataset path for specified framework and domain."""
    dataset = dataset_locations.get(Frameworks(framework), {}).get(Domains(domain), {})
    if dataset is None:
        raise Exception("Could not found dataset location.")
    if dataset.get("path") is None:
        raise Exception("Could not found dataset location.")
    return dataset["path"]


def get_framework_from_path(model_path: str) -> Optional[str]:
    """
    Get framework name from model extension.

    :param model_path: Path to model.
    """
    from neural_compressor.ux.components.model.repository import ModelRepository

    model_repository = ModelRepository()
    try:
        model = model_repository.get_model(model_path)
        return model.get_framework_name()
    except NotFoundException:
        return None


def get_file_extension(path: str) -> str:
    """Get file extension without leading dot."""
    return os.path.splitext(path)[1][1:]


def is_dataset_file(path: str) -> bool:
    """Check if given path is for a dataset of supported framework."""
    dataset_extensions = ["record", "tf_record"]

    extension = get_file_extension(path)

    return extension in dataset_extensions


def get_predefined_config_path(framework: str, domain: str, domain_flavour: str = "") -> str:
    """Get predefined config for specified model domain."""
    mapper = NamesMapper(MappingDirection.ToCore)
    mapped_framework = mapper.map_name("framework", framework)
    mapped_domain = mapper.map_name("domain", domain)
    mapped_domain_flavour = mapper.map_name("domain_flavour", domain_flavour)
    possible_filenames = [
        f"{mapped_domain}_{mapped_domain_flavour}.yaml",
        f"{mapped_domain}.yaml",
        "default.yaml",
    ]
    for filename in possible_filenames:
        config_path = os.path.join(
            configs_directory,
            "predefined_configs",
            f"{mapped_framework}",
            filename,
        )
        if config_path and os.path.isfile(config_path):
            return config_path
    raise NotFoundException(
        f"Could not found predefined config for {framework} {domain} {domain_flavour} model.",
    )


def normalize_domain(domain: str) -> str:
    """Normalize domain strings by replacing spaces to underscores and lowering case."""
    if domain == Domains.NLP.value:
        return "nlp"
    return domain.lower().replace(" ", "_")


def normalize_framework(framework: str) -> str:
    """Normalize framework strings by lowering case."""
    return framework.lower()


def normalize_string(string_to_normalize: str) -> str:
    """Normalize string for usage in path."""
    lowered_string = string_to_normalize.lower()
    return secure_filename(lowered_string)


def get_model_zoo_config_path(
    workspace_path: Optional[str],
    framework: str,
    domain: str,
    model_name: str,
    model_dict: Dict[str, Any],
) -> str:
    """Get predefined config for model from Examples."""
    if workspace_path is None:
        return ""
    model_dir = os.path.join(
        workspace_path,
        "examples",
        framework,
        domain,
        model_name,
    )
    yaml_relative_location = model_dict.get("yaml")
    if (
        model_dir
        and yaml_relative_location
        and os.path.exists(os.path.join(model_dir, yaml_relative_location))
    ):
        return os.path.join(model_dir, yaml_relative_location)
    return ""


def get_model_zoo_model_path(
    workspace_path: Optional[str],
    framework: str,
    domain: str,
    model_name: str,
    model_dict: Dict[str, Any],
) -> str:
    """Get path for model from Examples."""
    if workspace_path is None:
        return ""
    model_dir = os.path.join(
        workspace_path,
        "examples",
        framework,
        domain,
        model_name,
    )

    model_relative_path = model_dict.get("download", {}).get("filename", None)
    if (
        model_dir
        and model_relative_path
        and os.path.exists(os.path.join(model_dir, model_relative_path))
    ):
        return os.path.join(model_dir, model_relative_path)
    return ""


def _load_json_as_dict(path: str) -> dict:
    """Load json file and convert it into dict.

    :param path: path to json file
    :type path: str
    :return: dict
    :rtype: dict
    """
    with open(path, "r") as json_data:
        data = json.load(json_data)
    if isinstance(data, dict):
        return data
    return {}


def _load_json_as_list(path: str) -> list:
    """Load json file and convert it into list.

    :param path: path to json file
    :type path: str
    :return: dict
    :rtype: dict
    """
    with open(path, "r") as json_data:
        data = json.load(json_data)
    if isinstance(data, list):
        return data
    return []


def check_module(module_name: str) -> None:
    """Check if module exists. Raise exception when not found."""
    if module_name == "onnxrt":
        module_name = "onnxruntime"
    if module_name == "pytorch":
        module_name = "torch"
    module = find_spec(module_name.lower())
    if module is None:
        raise ClientErrorException(f"Could not find {module_name} module.")


def get_module_version(module_name: str) -> str:
    """Check module version. Raise exception when not found."""
    version = None
    if module_name == "onnxrt":
        module_name = "onnxruntime"
    command = [
        "python",
        "-c",
        f"import {module_name} as module; print(module.__version__)",
    ]
    proc = Proc()
    proc.run(args=command)
    if proc.is_ok:
        for line in proc.output:
            version = line.strip()
    proc.remove_logs()
    if version is None:
        raise ClientErrorException(f"Could not found version of {module_name} module.")
    return version


def get_size(path: str, unit: str = "MB", add_unit: bool = False) -> Union[str, int]:
    """Check file or directory size."""
    supported_units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
    }
    unit_modifier = supported_units.get(unit, None)
    if unit_modifier is None:
        raise Exception(
            "Unit not supported. Select one of following: " + str(supported_units.keys()),
        )
    root_dir = Path(path)
    if root_dir.is_file():
        size = root_dir.stat().st_size
    else:
        size = sum(f.stat().st_size for f in root_dir.glob("**/*") if f.is_file())
    size = int(round(size / unit_modifier))
    if add_unit:
        return f"{size}{unit}"
    return size


def load_model_config() -> Dict[str, Any]:
    """Load model configs from json."""
    json_path = os.path.join(
        configs_directory,
        "models.json",
    )
    return _load_json_as_dict(json_path)


def load_dataloader_config() -> List[Dict[str, Any]]:
    """Load dataloader configs from json."""
    json_path = os.path.join(
        configs_directory,
        "dataloaders.json",
    )
    return _load_json_as_list(json_path)


def load_transforms_config() -> List[Dict[str, Any]]:
    """Load transforms configs from json."""
    json_path = os.path.join(
        configs_directory,
        "transforms.json",
    )
    return _load_json_as_list(json_path)


def load_transforms_filter_config() -> Dict[str, Any]:
    """Load meaningful transforms configs from json."""
    json_path = os.path.join(
        configs_directory,
        "transforms_filter.json",
    )
    return _load_json_as_dict(json_path)


def load_precisions_config() -> dict:
    """Load transforms configs from json."""
    json_path = os.path.join(
        configs_directory,
        "precisions.json",
    )
    return _load_json_as_dict(json_path)


def load_model_wise_params(framework: str) -> dict:
    """Load model wise parameters for specified framework."""
    json_path = os.path.join(
        configs_directory,
        "model_wise_params.json",
    )
    model_wise_params = _load_json_as_dict(json_path)
    parameters = model_wise_params.get("common", {})
    framework_specific_params = model_wise_params.get(framework, {})
    for param_type, params_dict in framework_specific_params.items():
        for param in params_dict.keys():
            if param in parameters[param_type]:
                parameters[param_type][param].extend(params_dict[param])
            else:
                parameters[param_type].update({param: params_dict[param]})
    return {"model_wise": parameters}


def get_metrics_dict() -> dict:
    """Get metrics list from Neural Compressor and add help messages."""
    help_dict = load_help_nc_params("metrics")
    metrics = {}

    for framework, fw_metrics in registry_metrics.items():
        inc_framework_name = "onnxrt" if framework == "onnxrt_qlinearops" else framework
        if inc_framework_name.lower() not in [fw.value.lower() for fw in Frameworks]:
            continue
        raw_metric_list = list(fw_metrics.keys()) if fw_metrics else []
        raw_metric_list += ["custom"]
        metrics_updated = _update_metric_parameters(raw_metric_list)
        for metric, value in metrics_updated.copy().items():
            if isinstance(value, dict):
                for key in value.copy().keys():
                    for field in ["help", "label"]:
                        msg_key = f"__{field}__{key}"
                        metrics_updated[metric][msg_key] = help_dict.get(
                            metric,
                            {},
                        ).get(msg_key, "")
            metrics_updated[f"__help__{metric}"] = help_dict.get(
                f"__help__{metric}",
                "",
            )
        metrics.update({inc_framework_name: _parse_help_in_dict(metrics_updated)})

    return metrics


def _update_metric_parameters(metric_list: List[str]) -> Dict[str, Any]:
    """Add parameters to metrics."""
    metrics: Dict[str, Any] = {}
    for metric in metric_list:
        if metric == "topk":
            metrics.update({metric: {"k": [1, 5]}})
        elif metric == "COCOmAP":
            annotation_path = os.path.join(WORKDIR_LOCATION, "label_map.yaml")
            metrics.update({metric: {"anno_path": annotation_path}})
        elif metric in ["MSE", "RMSE", "MAE"]:
            metrics.update({metric: {"compare_label": True}})
        else:
            metrics.update({metric: None})
    return metrics


def _parse_help_in_dict(data: dict) -> list:
    parsed_list = []
    for key, value in data.items():
        if key.startswith("__help__") or key.startswith("__label__"):
            continue
        if isinstance(value, dict):
            parsed_list.append(
                {
                    "name": key,
                    "help": data.get(f"__help__{key}", ""),
                    "params": _parse_help_in_dict(value),
                },
            )
        else:
            item = {
                "name": key,
                "help": data.get(f"__help__{key}", ""),
                "value": value,
            }
            label = data.get(f"__label__{key}")
            if label:
                item["label"] = label
            parsed_list.append(item)
    return parsed_list


def load_help_nc_params(parameter: str) -> Dict[str, Any]:
    """Load help info from json for metrics, objectives and strategies."""
    json_path = os.path.join(
        configs_directory,
        f"{parameter}.json",
    )
    return _load_json_as_dict(json_path)


def replace_with_values(param: dict, file_path: str) -> None:
    """Replace parameters with value."""
    with open(file_path, "r+") as opened_file:
        text = opened_file.read()
        for key, value in param.items():
            key_to_search = "".join(["{{", key, "}}"])
            text = re.sub(key_to_search, value, text)
        opened_file.seek(0)
        opened_file.write(text)


def verify_file_path(path: str) -> None:
    """Check if path can be accessed."""
    restricted_paths = [
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/media",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/snap",
        "/srv",
        "/swapfile",
        "/usr",
        "/var",
    ]
    real_path = os.path.realpath(path)
    if not os.path.exists(real_path):
        raise NotFoundException("File not found.")
    if os.stat(real_path).st_uid == 0:
        raise AccessDeniedException("Access denied.")
    for path_element in real_path.split(os.sep):
        if path_element.startswith("."):
            raise AccessDeniedException("Access denied.")
    for restricted_path in restricted_paths:
        if real_path.startswith(restricted_path):
            raise AccessDeniedException("Access denied.")


def determine_ip() -> str:
    """Return IP to be used by server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.0.0.0", 1))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()

    return ip


def is_development_env() -> bool:
    """Return true if NC_MODE is development else false."""
    return os.environ.get("NC_MODE") == "development"


def filter_transforms(
    transforms: List[dict],
    framework: str,
    domain: str,
) -> List[dict]:
    """Collect only meaningful transform for specified framework and domain."""
    filtered_transforms = []
    transforms_filter = load_transforms_filter_config()
    domain_transforms = transforms_filter.get(framework, {}).get(domain, [])
    if len(domain_transforms) == 0:
        return transforms

    for transform in transforms:
        if transform.get("name") in domain_transforms:
            filtered_transforms.append(transform)

    return filtered_transforms


def parse_bool_value(value: Any) -> bool:
    """Parse value to boolean."""
    true_options = ["true", "t", "yes", "y", "1"]
    false_options = ["false", "f", "no", "n", "0"]
    if isinstance(value, str):
        value = value.lower().strip()
        if value in true_options:
            return True
        elif value in false_options:
            return False
        else:
            raise ValueError(
                f"Supported values boolean values are: "
                f"True ({true_options}), False ({false_options})",
            )
    return bool(value)


def release_tag() -> str:
    """Build tag based on release version."""
    version_pattern = r"^(?P<release>[0-9]+(\.[0-9]+)*).*?$"
    version_regex = re.compile(version_pattern)
    matches = version_regex.search(nc_version)

    if matches is None:
        raise ValueError(f"Unable to parse version {nc_version}")

    release_version = matches.groupdict().get("release")
    return f"v{release_version}"


def get_shape_from_transforms(
    transforms: List[Transform],
    shape_elements_order: List[str],
) -> list:
    """Detect dataset sizes based on configured transforms."""
    shapes = {
        "channels": 3,
        "height": None,
        "width": None,
    }

    for transform in transforms:
        name = transform.name
        parameters = transform.parameters
        if name in [
            "Resize",
            "CenterCrop",
            "RandomResizedCrop",
            "RandomCrop",
            "CropResize",
        ]:
            shapes["height"], shapes["width"] = get_height_width_from_size(parameters.get("size"))
        elif "Transpose" == name:
            axes = parameters.get("perm")
            if not axes:
                raise ValueError("Unknown value of 'perm' argument in Transpose")
            shape_elements_order = [shape_elements_order[idx] for idx in axes]
        elif "CropToBoundingBox" == name:
            shapes["height"] = parameters.get("target_height")
            shapes["width"] = parameters.get("target_width")
        elif name in [
            "ResizeCropImagenet",
            "BilinearImagenet",
        ]:
            shapes["height"] = parameters.get("height")
            shapes["width"] = parameters.get("width")

    if not shapes["height"] or not shapes["width"]:
        raise NotFoundException("Unable to detect shape for Dummy dataset")

    return [[shapes.get(dimension) for dimension in shape_elements_order]]


def get_height_width_from_size(size: Any) -> Tuple[Optional[int], Optional[int]]:
    """Detect dataset sizes based on size param common in some Transforms."""
    if isinstance(size, int):
        return size, size
    elif isinstance(size, list):
        if len(size) == 1:
            return size[0], size[0]
        elif len(size) == 2:
            return size[0], size[1]
    return None, None


def parse_to_string_list(values: Union[None, str, List[str]]) -> List[str]:
    """Parse to string list."""
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return values
    return []


def parse_to_float_list(values: Union[None, float, List[float]]) -> List[float]:
    """Parse to float list."""
    if isinstance(values, float):
        return [values]
    if isinstance(values, list):
        return values
    return []


def export_to_csv(data: List[dict], file: str) -> None:
    """Export list of dicts to csv file."""
    if len(data) < 1:
        raise InternalException("Could not export to csv. Data is empty.")
    with open(file, "w") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)
