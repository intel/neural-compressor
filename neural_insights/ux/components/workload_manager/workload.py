import os
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from neural_insights.ux.utils.consts import WorkloadMode, Frameworks
from neural_insights.ux.utils.exceptions import InternalException
from neural_insights.ux.utils.json_serializer import JsonSerializer
from neural_insights.ux.utils.utils import get_framework_from_path


class Workload(JsonSerializer):
    """Workload class."""

    def __init__(self, data: Optional[dict] = None):
        """Initialize Workload."""
        super().__init__()
        if data is None:
            data = {}
        self.uuid = str(data.get("uuid", uuid4()))
        self.creation_time: int = int(data.get("creation_time", datetime.now().timestamp()))
        self.workload_location: str = data.get("workload_location", None)

        mode = data.get("mode")
        if not isinstance(mode, WorkloadMode) and isinstance(mode, str):
            mode = WorkloadMode(mode)
        self.mode: WorkloadMode = mode

        self._model_path: str = data.get("model_path", None)

        framework = data.get("framework", None)
        if not isinstance(framework, Frameworks) and isinstance(framework, str):
            framework = Frameworks(get_framework_from_path(self.model_path))
        self.framework: Optional[Frameworks] = framework

    @property
    def model_path(self) -> str:
        """Get model_path."""
        return self._model_path

    @model_path.setter
    def model_path(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise InternalException("Could not locate model.")
        self._model_path = model_path
        self.framework = Frameworks(get_framework_from_path(self.model_path))

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Workload class."""
        return {
            "uuid": self.uuid,
            "framework": self.framework.value,
            "workload_location": self.workload_location,
            "mode": self.mode.value,
            "model_path": self.model_path,
            "creation_time": self.creation_time,
        }
