## INC Coding Conventions

 (Mostly for Version 3 and later)

### Goal

To improve the quality and maintainability of INC code, we summarized some common coding standards and conventions.
There are many popular programming conventions, and they may conflict with each other. To avoid overly arguing formatting, we make decisions based on the following priorities:

- [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings), [PEP 8](https://peps.python.org/pep-0008/)
- Framework Style
- INC Internal Style
- Sub-module specific Style

> Note: The sub-tile naming follows the [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings). For further information, go to the corresponding section.

### Import

- Recommend

```python
import os
import sys

from x import py
from x import y as z
from copy import deepcopy
from subprocess import Popen, PIPE
```

- Not recommend

```python
from modde import *  # May lead to namespace pollution
import os, sys  # Import on separate lines
import copy  # May import local copy.py
```



### Public Interface

Use `__all__` to help the developer and user know the supported interface and components.

```python
__all__ = [
    "register_config",
    "BaseConfig",
    "ComposableConfig",
    "get_all_config_set_from_config_registry",
    "options",
]
```



### Type Annotated Code

- Recommend

```python
def register_config(framework_name: str, algo_name: str, priority: int = 0) -> Callable[..., Any]:
    ...


eval_result: float = evaluator.eval(model)
```

- Not recommend

```python
def xx_func(cls) -> Dict[str, OrderedDict[str, Dict[str, object]]]: # Can't improve the readability
```

- Tools
  - python
  - pylance

### Logger

- Recommend

```python
from neural_compressor.common import logger

logger.info("Current TensorFlow Version is: %s", tf.__version__)  # Use a pattern-string (with %-placeholders)

logger.info("Current $PAGER is: %s", os.getenv("PAGER", default=""))  # Better readability
```

- Not recommend

```python
from neural_compressor.common.utils import info

info(
    "some log ..."
)  # The `filename` is always `logger.py`, like `2024-01-28 10:03:56 [INFO][logger.py:116] some log ...`

logger.info(f"Current TensorFlow Version is: {tf.__version__}")  # Not use f-string

logger.info("Current $PAGER is:")  # One sentence in two lines
logger.info(os.getenv("PAGER", default=""))
```



### Strings

- Recommend

```python
long_string = """This is fine if your use case can accept
	extraneous leading spaces."""

long_string = "And this is fine if you cannot accept\n" "extraneous leading spaces."
```



- Not recommend

```python
logger.info("This is fine if your use case can accept")
logger.info("extraneous leading spaces.")
```



### TODO Comments

- Recommend

```python
# TODO: crbug.com/192795 - Investigate cpufreq optimizations.
```

> A `TODO` comment begins with the word `TODO:` for more easily searchability.



### Comments

- Recommend

```python
# TODO: crbug.com/192795 - Investigate cpufreq optimizations.
# * Important information
# ? Need decision
# ! Deprecated method, do not use
```



- Recommend

```python
class CheeseShopAddress:
    """The address of a cheese shop.

    ...
    """


class OutOfCheeseError(Exception):
    """No more cheese is available."""
```

- Not recommend

```python
class CheeseShopAddress:
    """Class that describes the address of a cheese shop.

    ...
    """


class OutOfCheeseError(Exception):
    """Raised when no more cheese is available."""
```



### Folder structure

```shell
├── fwk_name
│   ├── __init__.py
│   ├── quantization
│   │   ├── algorithm_entry.py
│   │   ├── autotune.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   └── quantize.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── smooth_quant
│   │   │   ├── __init__.py
│   │   │   ├── smooth_quant.py
│   │   │   └── utility.py
│   │   ├── static_quant
│   │   │   ├── __init__.py
│   │   │   ├── static_quant.py
│   │   │   └── utility.py
│   │   └── weight_only
│   │       ├── gptq.py
│   │       ├── __init__.py
│   │       └── rtn.py
│   └── utils
│       ├── constants.py
│       ├── __init__.py
│       └── utility.py
└── __init__.py
```

```python
# * Note some code snippets
# neural_compressor/fwk_name/quantization/algorithm_entry.py
@register_algo(RTN)
def rtn_algo_entry()
    from neural_compressor.fwk_name.algorithms import rtn
    ...

@register_algo(SMOOTH_QUANT)
def smooth_quant_entry():
    from neural_compressor.fwk_name.algorithms import smooth_quant
    ...

```



### Module Tags

- Better categorize the PRs and issues.
- Tags list:`INC3.X`, `Auto-Tune`, `PyTorch`,`OnnxRuntime`,`Tensorflow`



### Recommend VS Code `settings.json`
- `settings.json` filepath: `neural-compressor/.vscode/settings.json`


### Reference

- [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
- [PEP 8](https://peps.python.org/pep-0008/)
- [PyTorch](https://github.com/pytorch/pytorch)
