# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module is only used as reference to convert Python docstring to API document.

The created the API document is in `API Doc`_.

Usage::

    $python api_doc_example.py

Example::

    def fun(a):
        return a+1

    x = fun(2)
    print(x)

Attributes:
    attribute1 (int): Module level attribute.

        Please set 1-100 integer.

Todo:
    * Improve the code format
    * Add more comments

.. _API Doc:
   https://intel.github.io/neural-compressor/latest/autoapi/neural_compressor/api_doc_example/index.html
"""

module_debug_level1 = 1
"""int: Module debug level document.

"""


def function1(param1, param2):
    """Example function1.

    Args:
        param1 (str): The parameter1.
        param2 (float): The parameter2.

    Example::

        >>> python api_doc_example.py
        >>> import os
        ... for i in range(3)
        ...     print(i)
        0
        1
        2

    Returns:
        bool: The return value. True|False.
    """


def function2(param1: str, param2: float) -> bool:
    """Function with PEP 484 type annotations.

    Args:
        param1: The parameter1.
        param2: The parameter2.

    Example:

        Style 3::

            from neural_compressor.config import MixedPrecisionConfig
            def eval_func(model):
                ...
                return accuracy

            conf = MixedPrecisionConfig()
            output_model = mix_precision.fit(
                model,
                conf,
                eval_func=eval_func,
            )

    Returns:
        The return value. True|False.
    """


def function3(param1, param2=None, *args, **kwargs):
    """This is an example of function3.

    If ``*args`` or ``**kwargs`` are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    Args:
        param1 (int): The parameter1.
        param2 (:obj:`str`, optional): The parameter2.
        *args: Arguments list.
        **kwargs: Key-value dict.

    Returns:
        bool: The return value. True|False.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of exceptions.
        ValueError: If `param2` is equal to `param1`.
    """
    if param1 == param2:
        raise ValueError("param1 may not be equal to param2")
    return True


def generator1(n):
    """Generators have a ``Yields`` section.

    Args:
        n (int): range.

    Yields:
        int: The next number in [0, `n` - 1].

    Examples::

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]
    """
    yield from range(n)


class ExampleClass:
    """Example for Class.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    """

    def __init__(self, param1, param2, param3):
        """Example of __init__ method.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (list(str)): Description of `param3`.
        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline*

        #: list(str): Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def property1(self):
        """str: Property is documented."""
        return "property1"

    def method1(self, param1, param2):
        """Method1 for execute.

        Note:
            It's public.

        Args:
            param1: The parameter1.
            param2: The parameter2.

        Returns:
            True|False.
        """
        return True

    def __special__(self):
        """This function won't be documented that start with and
        end with a double underscore."""
        pass

    def _private(self):
        """Private members are not included."""
        pass
