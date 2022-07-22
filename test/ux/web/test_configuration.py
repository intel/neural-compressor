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
"""UX Configuration test."""

import logging
import socket
import unittest
from unittest.mock import MagicMock, patch

from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION
from neural_compressor.ux.web.configuration import Configuration
from neural_compressor.ux.web.exceptions import NotFoundException


@patch("neural_compressor.ux.web.configuration.determine_ip", new=lambda: "127.0.0.1")
class TestConfiguration(unittest.TestCase):
    """UX Configuration tests."""

    @patch(
        "sys.argv",
        [
            "inc_bench.py",
        ],
    )
    @patch("secrets.token_hex")
    def test_defaults(
        self,
        mock_secrets_token_hex: MagicMock,
    ) -> None:
        """Test default values."""
        mock_secrets_token_hex.return_value = "this is a mocked token value"

        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(5000, configuration.server_port)
        self.assertEqual(5000, configuration.gui_port)
        self.assertEqual(logging.INFO, configuration.log_level)
        self.assertEqual("127.0.0.1", configuration.server_address)
        self.assertEqual("https", configuration.scheme)
        self.assertEqual("this is a mocked token value", configuration.token)
        self.assertEqual(
            "https://127.0.0.1:5000/?token=this is a mocked token value",
            configuration.get_url(),
        )

    @patch("sys.argv", ["inc_bench.py", "-P1234"])
    @patch("secrets.token_hex")
    def test_changing_gui_port(
        self,
        mock_secrets_token_hex: MagicMock,
    ) -> None:
        """Test changing GUI port."""
        mock_secrets_token_hex.return_value = "this is a mocked token value"

        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(1234, configuration.gui_port)
        self.assertNotEqual(configuration.server_port, configuration.gui_port)
        self.assertEqual(
            "https://127.0.0.1:1234/?token=this is a mocked token value",
            configuration.get_url(),
        )

    @patch("sys.argv", ["inc_bench.py", "-p1234"])
    @patch("secrets.token_hex")
    def test_changing_server_port(
        self,
        mock_secrets_token_hex: MagicMock,
    ) -> None:
        """Test changing API port."""
        mock_secrets_token_hex.return_value = "this is a mocked token value"

        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(1234, configuration.server_port)
        self.assertEqual(1234, configuration.gui_port)
        self.assertEqual(
            "https://127.0.0.1:1234/?token=this is a mocked token value",
            configuration.get_url(),
        )

    @patch("sys.argv", ["inc_bench.py", "-p 0"])
    def test_changing_server_port_too_low(self) -> None:
        """Test changing API port to invalid value."""
        configuration = Configuration()
        with self.assertRaisesRegex(
            ValueError,
            "Lowest allowed port number is 1, attempted to use: 0",
        ):
            configuration.set_up()

    @patch("sys.argv", ["inc_bench.py", "-p 65536"])
    def test_changing_server_port_too_high(self) -> None:
        """Test changing API port to invalid value."""
        configuration = Configuration()
        with self.assertRaisesRegex(
            ValueError,
            "Highest allowed port number is 65535, attempted to use: 65536",
        ):
            configuration.set_up()

    @patch("sys.argv", ["inc_bench.py", "-P 0"])
    def test_changing_gui_port_too_low(self) -> None:
        """Test changing GUI port to invalid value."""
        configuration = Configuration()
        with self.assertRaisesRegex(
            ValueError,
            "Lowest allowed port number is 1, attempted to use: 0",
        ):
            configuration.set_up()

    @patch("sys.argv", ["inc_bench.py", "-P 65536"])
    def test_changing_gui_port_too_high(self) -> None:
        """Test changing GUI port to invalid value."""
        configuration = Configuration()
        with self.assertRaisesRegex(
            ValueError,
            "Highest allowed port number is 65535, attempted to use: 65536",
        ):
            configuration.set_up()

    @patch("sys.argv", ["inc_bench.py", "-p1234", "-P5678"])
    @patch("secrets.token_hex")
    def test_changing_server_and_gui_port(
        self,
        mock_secrets_token_hex: MagicMock,
    ) -> None:
        """Test changing API and GUI ports."""
        mock_secrets_token_hex.return_value = "this is a mocked token value"

        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(1234, configuration.server_port)
        self.assertEqual(5678, configuration.gui_port)
        self.assertEqual(
            "https://127.0.0.1:5678/?token=this is a mocked token value",
            configuration.get_url(),
        )

    @patch("sys.argv", ["inc_bench.py", "-vv"])
    def test_changing_log_level_to_defined_one(self) -> None:
        """Test changing log level."""
        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(logging.DEBUG, configuration.log_level)

    @patch("sys.argv", ["inc_bench.py", "-vvvvvvvvvvvvv"])
    def test_changing_log_level_to_not_defined_one(self) -> None:
        """Test changing log level to unknown one."""
        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(logging.DEBUG, configuration.log_level)

    @patch("socket.socket.bind")
    @patch("sys.argv", ["inc_bench.py", "-p1234"])
    def test_changing_server_port_to_already_taken_fails(
        self,
        mock_socket_bind: MagicMock,
    ) -> None:
        """Test fail during attempting to use taken port."""
        mock_socket_bind.configure_mock(side_effect=socket.error)

        with self.assertRaises(NotFoundException):
            configuration = Configuration()
            configuration.set_up()

    @patch("socket.socket.bind")
    @patch("sys.argv", ["inc_bench.py"])
    def test_when_all_ports_taken_it_fails(
        self,
        mock_socket_bind: MagicMock,
    ) -> None:
        """Test fail when all ports taken."""
        mock_socket_bind.configure_mock(side_effect=socket.error)

        with self.assertRaises(NotFoundException):
            configuration = Configuration()
            configuration.set_up()

    @patch("sys.argv", ["inc_bench.py"])
    def test_many_instances_are_the_same(self) -> None:
        """Test that all instances references same object."""
        original_configuration = Configuration()
        new_configuration = Configuration()

        self.assertTrue(original_configuration is new_configuration)

    @patch("sys.argv", ["inc_bench.py"])
    def test_reloading_config_changes_token(self) -> None:
        """Test that reloading configuration changes token."""
        configuration = Configuration()
        original_token = configuration.token

        configuration.set_up()

        self.assertNotEqual(original_token, configuration.token)

    @patch("sys.argv", ["inc_bench.py"])
    def test_default_workdir(self) -> None:
        """Test that when no existing config given, default will be used."""
        configuration = Configuration()
        configuration.set_up()

        self.assertEqual(WORKSPACE_LOCATION, configuration.workdir)


if __name__ == "__main__":
    unittest.main()
