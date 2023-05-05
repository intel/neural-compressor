from typing import Optional, Dict, Any

from neural_insights.components.workload_manager.workload import Workload
from neural_insights.utils.json_serializer import JsonSerializer


class QuantizationWorkload(Workload):

    def __init__(self,  data: Optional[dict] = None):
        super().__init__(data)
        if data is None:
            data = {}
        self.accuracy_data = AccuracyData(data.get("accuracy_data", None))

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Workload class."""
        serialized_workload = super().serialize(serialization_type)
        serialized_workload.update({"accuracy_data": self.accuracy_data.serialize()})
        return serialized_workload


class AccuracyData(JsonSerializer):

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            data = {}
        self._baseline_accuracy: Optional[float] = data.get("baseline_accuracy", None)
        self._optimized_accuracy: Optional[float] = data.get("optimized_accuracy", None)

    @property
    def baseline_accuracy(self) -> Optional[float]:
        """
        Get baseline accuracy.

        Returns: baseline accuracy value
        """
        return self._baseline_accuracy

    @property
    def optimized_accuracy(self) -> Optional[float]:
        """
        Get optimized accuracy.

        Returns: optimized accuracy value
        """
        return self._optimized_accuracy

    @property
    def ratio(self) -> Optional[float]:
        """
        Get accuracy ratio.
        Returns: accuracy ratio if baseline and optimized accuracy are present
                 Otherwise returns None
        """
        if self.optimized_accuracy is None or self.baseline_accuracy is None:
            return None
        return (self.optimized_accuracy - self.baseline_accuracy) / self.baseline_accuracy

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize AccuracyData class."""
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "ratio": self.ratio,
        }
