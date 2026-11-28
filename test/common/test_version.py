"""Tests for the package version module.

``neural_compressor/common/version.py`` used to be a symbolic link pointing at
``neural_compressor/version.py``. Git for Windows defaults to
``core.symlinks=false`` and checks a symbolic link out as a plain text file
whose content is the link target, so ``import neural_compressor.common``
raised ``SyntaxError: invalid syntax`` on a default Windows checkout. The module
must stay a regular file; this test keeps it that way.
"""

import unittest
from pathlib import Path

import neural_compressor
from neural_compressor.common.version import __version__ as common_version
from neural_compressor.version import __version__ as package_version


class TestCommonVersion(unittest.TestCase):
    def test_common_version_matches_package_version(self):
        """The common module exposes the version declared in neural_compressor/version.py."""
        self.assertEqual(common_version, package_version)

    def test_common_version_module_is_a_regular_file(self):
        """A symlinked module cannot be imported on Windows checkouts (issue #2664)."""
        module_path = Path(neural_compressor.__file__).parent / "common" / "version.py"
        self.assertFalse(
            module_path.is_symlink(),
            f"{module_path} must be a regular file, not a symlink: Git for Windows turns it into "
            "a plain text file containing the link target",
        )


if __name__ == "__main__":
    unittest.main()
