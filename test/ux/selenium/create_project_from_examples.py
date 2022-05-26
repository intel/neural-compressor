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
"""Create project from examples test."""

import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--address", type=str, required=True)
parser.add_argument("--port", type=str, required=True)
parser.add_argument("--token", type=str, required=True)
args = parser.parse_args()

options = webdriver.ChromeOptions()
options.add_argument("--ignore-ssl-errors=yes")
options.add_argument("--ignore-certificate-errors")
driver = webdriver.Chrome(options=options)

driver.get(f"https://{args.address}:{args.port}/home?token={args.token}")

driver.implicitly_wait(0.5)

driver.find_element(By.ID, "create-new-project-btn").click()

driver.find_element(By.ID, "project_name").clear()
driver.find_element(By.ID, "project_name").send_keys("Test project")

WebDriverWait(driver, timeout=3).until(lambda d: d.find_element(By.ID, "domain0"))

driver.find_element(By.ID, "next_btn1").click()
driver.find_element(By.ID, "next_btn2").click()
driver.find_element(By.ID, "finish-basic-btn").click()

project_created = driver.find_element(By.ID, "project1").is_displayed()
assert project_created is True

driver.quit()
