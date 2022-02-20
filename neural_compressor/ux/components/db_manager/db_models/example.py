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
"""The Example class."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Example table")


class Example(Base):
    """INC Bench examples' table representation."""

    __tablename__ = "example"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    framework = Column(Integer, ForeignKey("framework.id"), nullable=False)
    domain = Column(Integer, ForeignKey("domain.id"), nullable=False)
    dataset_type = Column(String(50), nullable=False)
    model_url = Column(String(250), nullable=False)
    config_url = Column(String(250), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
