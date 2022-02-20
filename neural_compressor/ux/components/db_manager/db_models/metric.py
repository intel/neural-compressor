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
# pylint: disable=no-member
"""Metric package contains metric class representing metric table in database."""
import json
from typing import Any, List

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, event
from sqlalchemy.orm import Query, relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import get_metrics_dict

log.debug("Initializing Metric table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class Metric(Base):
    """INC Bench Metric table class."""

    __tablename__ = "metric"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    help = Column(String(250))
    params = Column(String)
    framework_id = Column(Integer, ForeignKey("framework.id"))
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    framework: Any = relationship("Framework", primaryjoin="Framework.id == Metric.framework_id")

    __table_args__ = (
        UniqueConstraint(
            "name",
            "framework_id",
            name="_metric_framework_name_uc",
        ),
    )

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available metrics."""
        metric_instances = db_session.query(Metric).order_by(Metric.id)
        metrics = Metric.query_to_list(metric_instances)
        return {"metrics": metrics}

    @staticmethod
    def list_by_framework(db_session: session.Session, framework_id: int) -> dict:
        """List available metrics."""
        metric_instances = (
            db_session.query(Metric)
            .order_by(Metric.id)
            .filter(Metric.framework_id == framework_id)
        )
        metrics = Metric.query_to_list(metric_instances)
        return {"metrics": metrics}

    @staticmethod
    def query_to_list(metrics_query: Query) -> List[dict]:
        """Convert query to list."""
        metrics_list = []
        for metric in metrics_query:
            metrics_list.append(
                {
                    "id": metric.id,
                    "name": metric.name,
                    "help": metric.help,
                    "params": json.loads(metric.params),
                    "framework": {
                        "id": metric.framework.id,
                        "name": metric.framework.name,
                    },
                },
            )
        return metrics_list


@event.listens_for(Metric.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for framework, fw_metrics in get_metrics_dict().items():
            framework_id = Framework.get_framework_id(db_session, framework_name=framework)
            if framework_id is None:
                raise InternalException("Could not found framework in database.")
            for metric in fw_metrics:
                db_session.add(
                    Metric(
                        name=metric.get("name"),
                        help=metric.get("help", ""),
                        params=json.dumps(metric.get("params", [])),
                        framework_id=framework_id,
                    ),
                )
