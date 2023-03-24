import json
import os
from os import PathLike
from typing import List, Optional

from neural_insights.ux.components.workload_manager.workload import Workload
from neural_insights.ux.utils.consts import WORKDIR_LOCATION
from neural_insights.ux.utils.exceptions import InternalException, ClientErrorException
from neural_insights.ux.utils.json_serializer import JsonSerializer
from neural_insights.ux.utils.logger import log
from neural_insights.ux.utils.singleton import Singleton


class WorkloadManager(JsonSerializer, metaclass=Singleton):
    """Workload Manager class."""

    def __init__(self, workdir_location: Optional[PathLike] = None):
        super().__init__()
        if workdir_location is None:
            log.debug("Using default workdir location.")
            workdir_location = WORKDIR_LOCATION
        self.config_path = os.path.join(workdir_location, "neural_insights", "config.json")
        self._version: int = 1
        self._workloads: List[Workload] = []

        self.load_workloads()

    def add_workload(self, workload: Workload) -> None:
        """Add workload."""
        self._workloads.append(workload)
        self.dump_config()

    def load_workloads(self) -> None:
        """Read workloads from config file."""
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "r") as config_file:
            config_data = json.load(config_file)
        config_version = int(config_data.get("version"))
        if config_version != self._version:
            raise InternalException(
                f"Incompatible config version has been found. "
                f"Expected version: {self._version}, found {config_version}.",
            )
        workloads_config = config_data.get("workloads", [])
        if not isinstance(workloads_config, list):
            raise InternalException(
                "Incompatible format of workloads. Workloads should be a list.",
            )
        self._workloads = []
        for workload_config in workloads_config:
            workload = Workload(workload_config)
            self._workloads.append(workload)

    def dump_config(self) -> None:
        """Dump workloads to config file."""
        serialized_workloads = [workload.serialize() for workload in self._workloads]
        data = {
            "version": self._version,
            "workloads": serialized_workloads,
        }

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as config_file:
            json.dump(data, config_file)

    @property
    def workloads(self) -> List[Workload]:
        """Get workloads list."""
        self.load_workloads()
        return self._workloads

    def get_workload(self, workload_uuid: str) -> Workload:
        """Get workload from workloads list."""
        for workload in self.workloads:
            if workload.uuid == workload_uuid:
                return workload
        raise ClientErrorException("Could not find workload with specified ID.")
