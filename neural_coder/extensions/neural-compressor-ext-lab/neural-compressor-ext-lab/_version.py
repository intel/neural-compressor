# Copyright (c) 2022 Intel Corporation
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


__all__ = ["__version__"]


def _fetchVersion():
    import json
    import os
    from pathlib import Path

    HERE = os.path.abspath(os.path.dirname(__file__))
    # HERE = Path(__file__).parent

    for d, _, _ in os.walk(HERE):
        print("walk dir",d)
        try:
            with open(os.path.join(d, "package.json")) as f:
                return json.load(f)["version"]
        except FileNotFoundError:
            pass

    raise FileNotFoundError("Could not find package.json under dir {}".format(HERE))


__version__ = _fetchVersion()
