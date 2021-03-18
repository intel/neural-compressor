# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
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
"""HWInfo test."""

import unittest
from unittest.mock import MagicMock, patch

from lpot.ux.utils.hw_info import HWInfo


class TestHWInfo(unittest.TestCase):
    """HWInfo tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Hardware Info tests constructor."""
        super().__init__(*args, **kwargs)

    @patch("psutil.cpu_count")
    def test_cores_num(self, mock_cpu_count: MagicMock) -> None:
        """Test if hw info uses psutil cpu_count to get number of cores."""
        mock_cpu_count.return_value = 8

        hw_info = HWInfo()
        self.assertEqual(hw_info.cores, 8)

    @patch("subprocess.Popen")
    def test_sockets_num(self, mock_subprocess: MagicMock) -> None:
        """Test getting number of sockets."""
        mock_subprocess.return_value.stdout = [b"           4\n"]

        hw_info = HWInfo()
        self.assertEqual(hw_info.sockets, 4)

    @patch("platform.release")
    @patch("platform.system")
    @patch("psutil.LINUX", False)
    @patch("psutil.WINDOWS", True)
    def test_get_windows_distribution(
        self,
        mock_platform_system: MagicMock,
        mock_platform_release: MagicMock,
    ) -> None:
        """Test getting windows system distribution."""
        mock_platform_system.return_value = "Windows"
        mock_platform_release.return_value = "10"

        hw_info = HWInfo()
        self.assertEqual(hw_info.system, "Windows 10")

    @patch("platform.dist")
    @patch("psutil.LINUX", True)
    @patch("psutil.WINDOWS", False)
    def test_get_linux_distribution(
        self,
        mock_platform_dist: MagicMock,
    ) -> None:
        """Test getting linux system distribution."""
        mock_platform_dist.return_value = (
            "DistroName",
            "DistroVerID",
            "DistroVerCodename",
        )

        hw_info = HWInfo()
        self.assertEqual(hw_info.system, "DistroName DistroVerID DistroVerCodename")

    @patch("platform.release")
    @patch("platform.system")
    @patch("platform.dist")
    @patch("psutil.LINUX", True)
    @patch("psutil.WINDOWS", False)
    def test_get_linux_distribution_without_dist(
        self,
        mock_platform_dist: MagicMock,
        mock_platform_system: MagicMock,
        mock_platform_release: MagicMock,
    ) -> None:
        """Test getting linux system distribution."""
        mock_platform_dist.configure_mock(side_effect=AttributeError)
        mock_platform_system.return_value = "Linux"
        mock_platform_release.return_value = "kernel_ver-88-generic"

        hw_info = HWInfo()
        self.assertEqual(hw_info.system, "Linux kernel_ver-88-generic")

    @patch("platform.platform")
    @patch("psutil.LINUX", False)
    @patch("psutil.WINDOWS", False)
    def test_get_unknown_os_distribution(
        self,
        mock_platform_platform: MagicMock,
    ) -> None:
        """Test getting unknown system distribution."""
        mock_platform_platform.return_value = "SystemName-Version-Arch"

        hw_info = HWInfo()
        self.assertEqual(hw_info.system, "SystemName-Version-Arch")


if __name__ == "__main__":
    unittest.main()
