#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2022 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""ONNX Operator Schemas for Tensorflow model converting to ONNX model."""

import logging
from collections import OrderedDict, defaultdict

from onnx import defs

from . import tf2onnx_utils as utils

logger = logging.getLogger("neural_compressor")


class OnnxOpSchema(object):
    """Wrapper for Onnx schema."""

    def __init__(self, name, domain, since_version, attributes):
        """Create a Onnx schema.

        Args:
            name (str): op name
            domain (str): default value "" means it's Onnx domain
            since_version (int): opset version, default is 1
            attributes (List[str]): valid attributes
        """
        self._name = name
        self._domain = domain
        self._attributes = attributes
        self._since_version = since_version

    @property
    def attributes(self):
        """Get valid attributes."""
        return self._attributes

    @property
    def domain(self):
        """Get domain info."""
        return self._domain

    @property
    def name(self):
        """Get op name."""
        return self._name

    @property
    def since_version(self):
        """Get opset version."""
        return self._since_version

    @staticmethod
    def from_onnx_schema(onnx_schema):
        """Static method to construct OnnxOpSchema."""
        name = onnx_schema.name
        domain = onnx_schema.domain
        since_version = int(onnx_schema.since_version)
        attributes = onnx_schema.attributes
        return OnnxOpSchema(name, domain, since_version, attributes)

    def has_attribute(self, attr):
        """Check if has the attribute."""
        return attr in self.attributes


def _register_all_schemas_with_history():
    """Register all schemas with history."""
    onnx_schemas = defs.get_all_schemas_with_history()
    name_domain_version_schema_map = defaultdict(lambda: defaultdict(dict))
    for s in onnx_schemas:
        schema = OnnxOpSchema.from_onnx_schema(s)
        name_domain_version_schema_map[schema.name][schema.domain][schema.since_version] = schema

    ordered_map = defaultdict(lambda: defaultdict(OrderedDict))
    for name, domain_version_schema_map in name_domain_version_schema_map.items():
        for domain, version_schema_map in domain_version_schema_map.items():
            ordered_map[name][domain] = OrderedDict(sorted(version_schema_map.items(), key=lambda x: -x[0]))
    return ordered_map


def _parse_domain_opset_versions(schemas):
    """Get max opset version among all schemas within each domain."""
    domain_opset_versions = dict()
    for domain_version_schema_map in schemas.values():
        for domain, version_schema_map in domain_version_schema_map.items():
            # version_schema_map is sorted by since_version in descend order
            max_version = next(iter(version_schema_map))
            if domain not in domain_opset_versions:
                domain_opset_versions[domain] = int(max_version)
            else:
                domain_opset_versions[domain] = max(domain_opset_versions[domain], int(max_version))
    return domain_opset_versions


# format is <OpName, <Domain, <SinceVersion, OpSchema>>>
# SinceVersion is sorted from high to low
_schemas = _register_all_schemas_with_history()

_domain_opset_versions = _parse_domain_opset_versions(_schemas)


def get_schema(name, max_inclusive_opset_version, domain=None):
    """Get schema by name within specific version."""
    domain = domain or utils.ONNX_DOMAIN
    domain_version_schema_map = _schemas[name]
    version_schema_map = domain_version_schema_map[domain]
    for version, schema in version_schema_map.items():
        if version <= max_inclusive_opset_version:
            return schema
    return None


def get_max_supported_opset_version(domain=None):
    """Get max supported opset version by current onnx package given a domain."""
    domain = domain or utils.ONNX_DOMAIN
    return _domain_opset_versions.get(domain, None)
