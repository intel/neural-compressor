# -*- coding: utf-8 -*-
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
"""INC Bench create project tests."""

import os
import time
from test.ux.selenium.utils import get_number_of_projects

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


class TestCreateProject:
    """Test creating project."""

    def setup_method(self, method):
        """Set up chromedriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--ignore-ssl-errors=yes")
        options.add_argument("--ignore-certificate-errors")

        self.driver = webdriver.Chrome(options=options)
        self.vars = {}

    def teardown_method(self, method):
        """Clean up."""
        self.driver.quit()

    def test_project_with_custom_model(self, params):
        """Test creating project with custom model."""
        address = params["address"]
        port = params["port"]
        url_prefix = params["url_prefix"]
        token = params["token"]
        inc_url = f"https://{address}:{port}/{url_prefix}/home?token={token}"

        models_dir = params["models_dir"]
        model_path = os.path.join(models_dir, "resnet50_v1_5.pb")

        self.driver.get(inc_url)

        initial_number_of_projects = get_number_of_projects(self.driver)

        self.driver.find_element(
            By.CSS_SELECTOR,
            "#create-new-project-menu-btn > .mat-button-wrapper",
        ).click()

        self.driver.find_element(By.ID, "project_name").click()

        element = self.driver.find_element(By.ID, "project_name")
        actions = ActionChains(self.driver)
        actions.double_click(element).perform()

        element = self.driver.find_element(By.ID, "next_btn1")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()

        self.driver.find_element(By.ID, "project_name").clear()
        self.driver.find_element(By.ID, "project_name").send_keys("Custom model project")

        self.driver.find_element(By.ID, "next_btn1").click()

        WebDriverWait(self.driver, 10).until(
            expected_conditions.element_to_be_clickable(
                (By.CSS_SELECTOR, "#custom-radio .mat-radio-outer-circle"),
            ),
        )

        element = self.driver.find_element(
            By.CSS_SELECTOR,
            "#custom-radio .mat-radio-outer-circle",
        )
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()

        element = self.driver.find_element(By.ID, "next_btn2")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()

        self.driver.find_element(By.ID, "next_btn2").click()

        self.driver.find_element(By.ID, "model_path").click()

        self.driver.find_element(By.ID, "model_path").send_keys(model_path)

        self.driver.get_screenshot_as_file("custom_model_path_given.png")

        try:
            WebDriverWait(self.driver, 30).until(
                expected_conditions.element_to_be_clickable(
                    (By.CSS_SELECTOR, '#finish-adv-btn:not([disabled="true"])'),
                ),
            )
        except Exception as err:
            self.driver.get_screenshot_as_file("custom_model_finish_button_wait_failed.png")
            raise err

        self.driver.find_element(By.ID, "finish-adv-btn").click()

        time.sleep(3)

        self.driver.get_screenshot_as_file("custom_model_project_added.png")

        final_number_of_projects = get_number_of_projects(self.driver)
        assert initial_number_of_projects + 1 == final_number_of_projects

    def test_project_from_examples(self, params):
        """Test creating project from examples."""
        address = params["address"]
        port = params["port"]
        url_prefix = params["url_prefix"]
        token = params["token"]
        inc_url = f"https://{address}:{port}/{url_prefix}/home?token={token}"

        self.driver.get(inc_url)

        initial_number_of_projects = get_number_of_projects(self.driver)

        self.driver.find_element(
            By.CSS_SELECTOR,
            "#create-new-project-menu-btn > .mat-button-wrapper",
        ).click()

        self.driver.find_element(By.ID, "project_name").click()

        element = self.driver.find_element(By.ID, "project_name")
        actions = ActionChains(self.driver)
        actions.double_click(element).perform()

        element = self.driver.find_element(By.ID, "next_btn1")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()

        self.driver.find_element(By.ID, "project_name").clear()
        self.driver.find_element(By.ID, "project_name").send_keys("Examples - mobilenet_v1")

        self.driver.find_element(By.ID, "next_btn1").click()

        element = self.driver.find_element(By.ID, "next_btn2")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).perform()

        self.driver.find_element(By.ID, "next_btn2").click()

        WebDriverWait(self.driver, timeout=30).until(lambda d: d.find_element(By.ID, "domain0"))

        self.driver.find_element(By.CSS_SELECTOR, "#model2 > .mat-button-wrapper").click()

        WebDriverWait(self.driver, 30).until(
            expected_conditions.element_to_be_clickable(
                (By.CSS_SELECTOR, '#finish-basic-btn:not([disabled="true"])'),
            ),
        )

        self.driver.find_element(By.ID, "finish-basic-btn").click()

        time.sleep(3)

        self.driver.get_screenshot_as_file("example_model_project_added.png")

        final_number_of_projects = get_number_of_projects(self.driver)
        assert initial_number_of_projects + 1 == final_number_of_projects
