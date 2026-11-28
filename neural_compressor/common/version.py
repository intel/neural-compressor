# Copyright (c) 2026 Intel Corporation
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
"""Expose the package version declared in :mod:`neural_compressor.version`.

Keep this module a regular file. Git for Windows defaults to
``core.symlinks=false``, which checks a symbolic link out as a plain text file
whose content is the link target; a symlinked module therefore makes
``import neural_compressor.common`` raise ``SyntaxError`` on every default
Windows checkout.
"""

from neural_compressor.version import __version__

__all__ = ["__version__"]
