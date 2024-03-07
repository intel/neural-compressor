from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from neural_compressor.common import logger


class _Evaluator:
    """_Evaluator is a collection of evaluation functions.

    Examples:
        def eval_acc(model):
            ...

        def eval_perf(molde):
            ...

        # Usage
        user_eval_fns1 = eval_acc
        user_eval_fns2 = {"eval_fn": eval_acc}
        user_eval_fns3 = {"eval_fn": eval_acc, "weight": 1.0, "name": "accuracy"}
        user_eval_fns4 = [
            {"eval_fn": eval_acc, "weight": 0.5},
            {"eval_fn": eval_perf, "weight": 0.5, "name": "accuracy"},
            ]
    """

    EVAL_FN = "eval_fn"
    WEIGHT = "weight"
    FN_NAME = "name"
    EVAL_FN_TEMPLATE: Dict[str, Any] = {EVAL_FN: None, WEIGHT: 1.0, FN_NAME: None}

    def __init__(self) -> None:
        self.eval_fn_registry: List[Dict[str, Any]] = []

    def evaluate(self, model) -> float:
        """Evaluate the model using registered evaluation functions.

        Args:
            model: The fp32 model or quantized model.

        Returns:
            The overall result of all registered evaluation functions.
        """
        result = 0
        for eval_pair in self.eval_fn_registry:
            eval_fn = eval_pair[self.EVAL_FN]
            eval_result = eval_fn(model)
            result = self._update_the_objective_score(eval_pair, eval_result, result)
        return result

    def _update_the_objective_score(self, eval_pair, eval_result, overall_result) -> float:
        return overall_result + eval_result * eval_pair[self.WEIGHT]

    def get_number_of_eval_functions(self) -> int:
        return len(self.eval_fn_registry)

    def _set_eval_fn_registry(self, user_eval_fns: List[Dict]) -> None:
        self.eval_fn_registry = [
            {
                self.EVAL_FN: user_eval_fn_pair[self.EVAL_FN],
                self.WEIGHT: user_eval_fn_pair.get(self.WEIGHT, 1.0),
                self.FN_NAME: user_eval_fn_pair.get(self.FN_NAME, user_eval_fn_pair[self.EVAL_FN].__name__),
            }
            for user_eval_fn_pair in user_eval_fns
        ]

    def set_eval_fn_registry(self, eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> None:
        # About the eval_fns format, refer the class docstring for details.
        if eval_fns is None:
            return
        elif callable(eval_fns):
            # single eval_fn
            eval_fn_pair = deepcopy(self.EVAL_FN_TEMPLATE)
            eval_fn_pair[self.EVAL_FN] = eval_fns
            eval_fn_pair[self.FN_NAME] = eval_fns.__name__
            eval_fns = [eval_fn_pair]
        elif isinstance(eval_fns, Dict):
            eval_fns = [eval_fns]
        elif isinstance(eval_fns, List):
            assert all([isinstance(eval_fn_pair, Dict) for eval_fn_pair in eval_fns])
        else:
            raise NotImplementedError(f"The eval_fns should be a dict or a list of dict, but got {type(eval_fns)}.")
        self._set_eval_fn_registry(eval_fns)

    def self_check(self) -> None:
        # check the number of evaluation functions
        num_eval_fns = self.get_number_of_eval_functions()
        assert num_eval_fns > 0, "Please ensure that you register at least one evaluation metric for auto-tune."
        logger.info("There are %d evaluations functions.", num_eval_fns)


def create_evaluator_for_eval_fns(eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> _Evaluator:
    evaluator = _Evaluator()
    evaluator.set_eval_fn_registry(eval_fns)
    return evaluator
