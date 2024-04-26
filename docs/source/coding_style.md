# Coding Conventions


## Background

To improve the quality and maintainability of INC code, we summarized some common coding standards and conventions.

There are many style guides, and they may conflict with each other. To avoid overly arguing formatting, we make decisions based on the following priorities:

- [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings), [PEP 8](https://peps.python.org/pep-0008/)
- Framework Style
- Internal Style
- Sub-module specific Style

## Rules

> Note: The sub-tile naming is following [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) and [PEP 8](https://peps.python.org/pep-0008/). See the relevant section for more details.


### Imports

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
from sub_module import *  # May lead to namespace pollution
import os, sys  # Import on separate lines
import copy  # May import local copy.py
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

### Logger

- Recommend

```python
from neural_compressor_ort import logger

logger.info("Current TensorFlow Version is: %s", tf.__version__)  # Use a pattern-string (with %-placeholders)

logger.info("Current $PAGER is: %s", os.getenv("PAGER", default=""))  # Better readability

# Handle long string
logger.warning(
    "All tuning options for the current strategy have been tried. \n"
    "If the quantized model does not seem to work well, it might be worth considering other strategies."
)

logger.warning(
    "This is a long string, this is a long string,"
    "override the user config's smooth quant alpha into the best alpha(%.4f) found in pre-strategy.",
    0.65421,
)
```

- Not recommend

```python
logger.info(f"Current ONNX Runtime Version is: {ort.__version__}")  # Use f-string

logger.info("Current $PAGER is:")  # One sentence in two lines
logger.info(os.getenv("PAGER", default=""))
```


### Type Annotations

- Recommend

```python
def register_config(framework_name: str, algo_name: str, priority: int = 0) -> Callable[..., Any]: ...


eval_result: float = evaluator.eval(model)

# Declare aliases of complex types
from typing import TypeAlias

ComplexTFMap: TypeAlias = Mapping[str, _LossAndGradient]
```

- Not recommend

```python
def xx_func(cls) -> Dict[str, OrderedDict[str, Dict[str, object]]]: # Can't improve the readability
```

- Plugs:
  - [python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)


### Comments

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


### TODO Comments

- Recommend

```python
# TODO: crbug.com/192795 - Investigate cpufreq optimizations.

# * Important information
# ? Need decision
# ! Deprecated method, do not use
```

> A `TODO` comment begins with the word `TODO:` for facilitate searching.

- Plug:
    [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments)


### Public and Internal Interfaces

Use `__all__` to help the developer and user know the supported interface and components.

```python
__all__ = [
    "options",
    "register_config",
    "get_all_config_set_from_config_registry",
    "BaseConfig",
]
```

## Recommend VS Code `settings.json`
To keep the coding style consistent, we suggest you replace `.vscode/settings.json` with `neural-compressor/.vscode/settings_recommended.json`.


## Reference

- [Google Python Style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
- [PEP 8](https://peps.python.org/pep-0008/)
