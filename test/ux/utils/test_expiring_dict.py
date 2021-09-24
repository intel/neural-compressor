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
"""ExpiringDict test."""
import time
import unittest

from neural_compressor.ux.utils.expiring_dict import ExpiringDict, ExpiringDictItem


class TestExpiringDictItem(unittest.TestCase):
    """ExpiringDictItem tests."""

    def test_it_sets_values(self) -> None:
        """Test if values are set correctly."""
        value = "this is expected value"
        expires_at = time.time() + 2

        item = ExpiringDictItem(value=value, expires_at=expires_at)

        self.assertEqual(value, item.value)
        self.assertEqual(expires_at, item.expires_at)

    def test_it_expires_after_time_passed(self) -> None:
        """Test if ExpiringDictItem expires after given time."""
        value = "this is expected value"
        expires_at = time.time() + 2

        item = ExpiringDictItem(value=value, expires_at=expires_at)

        self.assertFalse(item.is_expired())

        time.sleep(3)
        self.assertTrue(item.is_expired())


class TestExpiringDict(unittest.TestCase):
    """ExpiringDict tests."""

    def test_items_disappear_after_expiration(self) -> None:
        """Test that item is not accessible after expiration."""
        key = "tested key"
        value = "expected value"

        items = ExpiringDict(ttl=2)

        with self.assertRaisesRegex(KeyError, expected_regex=key):
            items[key]

        items[key] = value

        self.assertEqual(value, items[key])

        time.sleep(3)

        with self.assertRaisesRegex(KeyError, expected_regex=key):
            items[key]

    def test_items_disappear_after_expiration_when_passed_in_constructor(self) -> None:
        """Test that item is not accessible after expiration."""
        key = "tested key"
        value = "expected value"

        items = ExpiringDict(
            initial_value={
                key: value,
            },
            ttl=2,
        )

        self.assertEqual(value, items[key])

        time.sleep(3)

        with self.assertRaisesRegex(KeyError, expected_regex=key):
            items[key]


if __name__ == "__main__":
    unittest.main()
