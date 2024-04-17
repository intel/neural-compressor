import os
import habana_frameworks.torch.core as htcore
from typing import Optional
from .._quant_common.quant_config import Fp8cfg, QuantMode
from .._core import prepare_model
from .._core.measure import save_measurements
from .._core import prepare_model
from .._core.measure import save_measurements
from .._quant_common.quant_config import _read_config_from_file, Fp8cfg, get_hqt_config, set_hqt_config


def _prep_model_with_predefined_config(model, *, config: Fp8cfg):
    set_hqt_config(model, config)
    prepare_model(model)


def prep_model(model, config_path: Optional[str] = None):
    """
    Prepare this model with the given (absolute or relative) path of the json file containing the configuration.
    If `config_path` is not given or `None`,
    instead perform the legacy behavior of checking for env variable `QUANT_CONFIG`.
    """
    htcore.hpu_initialize()

    if config_path is None:
        config_path = os.getenv("QUANT_CONFIG")
        if config_path is None:
            raise EnvironmentError(
                "Either pass config_path parameter explicitly (recommended), or set environment variable QUANT_CONFIG"
            )

    config = _read_config_from_file(config_path=config_path)
    config = Fp8cfg.parse(config)
    return _prep_model_with_predefined_config(model, config=config)


def finish_measurements(model):
    save_measurements(model)
    print("Dumping measurements")


