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
"""Dataloader package contains dataloader class representing dataloader table in database."""
import json
from typing import Any, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import Query, relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import load_dataloader_config

log.debug("Initializing Dataloader table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class Dataloader(Base):
    """INC Bench Dataloader table class."""

    __tablename__ = "dataloader"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    help = Column(String(250))
    show_dataset_location = Column(Boolean, default=False)
    params = Column(String)
    framework_id = Column(Integer, ForeignKey("framework.id"))
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    framework: Any = relationship(
        "Framework",
        primaryjoin="Framework.id == Dataloader.framework_id",
    )

    __table_args__ = (
        UniqueConstraint(
            "name",
            "framework_id",
            name="_dataloader_framework_name_uc",
        ),
    )

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available dataloaders."""
        dataloader_instances = db_session.query(Dataloader).order_by(Dataloader.id)
        dataloaders = Dataloader.query_to_list(dataloader_instances)
        return {"dataloaders": dataloaders}

    @staticmethod
    def list_by_framework(db_session: session.Session, framework_id: int) -> dict:
        """List available dataloaders."""
        dataloader_instances = (
            db_session.query(Dataloader)
            .order_by(Dataloader.id)
            .filter(Dataloader.framework_id == framework_id)
        )
        dataloaders = Dataloader.query_to_list(dataloader_instances)
        return {"dataloaders": dataloaders}

    @staticmethod
    def query_to_list(dataloaders_query: Query) -> List[dict]:
        """Convert query to list."""
        dataloaders_list = []
        for dataloader in dataloaders_query:
            dataloaders_list.append(
                {
                    "id": dataloader.id,
                    "name": dataloader.name,
                    "help": dataloader.help,
                    "show_dataset_location": dataloader.show_dataset_location,
                    "params": json.loads(dataloader.params),
                    "framework": {
                        "id": dataloader.framework.id,
                        "name": dataloader.framework.name,
                    },
                },
            )
        return dataloaders_list

    @staticmethod
    def update_params(
        db_session: session.Session,
        dataloader_id: int,
        params: List[dict],
    ) -> dict:
        """Update dataloader default parameters."""
        dataloader = db_session.query(Dataloader).filter(Dataloader.id == dataloader_id).one()
        dataloader.params = json.dumps(params)
        db_session.add(dataloader)
        db_session.flush()

        return {
            "id": dataloader.id,
            "params": dataloader.params,
        }


@event.listens_for(Dataloader.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for fw_dataloader in load_dataloader_config():
            framework = fw_dataloader.get("name")
            if framework is None:
                raise InternalException("Could not find framework name in dataloaders' json.")
            framework_id = Framework.get_framework_id(db_session, framework_name=framework)
            if framework_id is None:
                raise InternalException("Could not found framework in database.")
            for dataloader in fw_dataloader.get("params", []):
                db_session.add(
                    Dataloader(
                        name=dataloader.get("name"),
                        help=dataloader.get("help"),
                        show_dataset_location=dataloader.get("show_dataset_location"),
                        params=json.dumps(dataloader.get("params", [])),
                        framework_id=framework_id,
                    ),
                )
