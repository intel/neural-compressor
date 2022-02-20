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
"""DomainFlavour package contains DomainFlavour class representing domain_flavour table in db."""
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, event
from sqlalchemy.orm import relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.utils.consts import DomainFlavours
from neural_compressor.ux.utils.logger import log

log.debug("Initializing DomainFlavour table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class DomainFlavour(Base):
    """INC Bench DomainFlavour table class."""

    __tablename__ = "domain_flavour"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    models: Any = relationship("Model", back_populates="domain_flavour")

    @staticmethod
    def get_domain_flavour_id(db_session: session.Session, domain_flavour_name: str) -> int:
        """Get domain_flavour id from name."""
        domain_flavour_ids = db_session.query(DomainFlavour.id).filter(
            DomainFlavour.name == domain_flavour_name,
        )
        if domain_flavour_ids.count() != 1:
            raise Exception("Domain flavour name is not unique.")
        return domain_flavour_ids[0].id

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available domain flavours."""
        domain_flavours = []
        domain_flavour_instances = db_session.query(
            DomainFlavour.id,
            DomainFlavour.name,
        ).order_by(DomainFlavour.id)
        for domain_flavour in domain_flavour_instances:
            domain_flavours.append(
                {
                    "id": domain_flavour.id,
                    "name": domain_flavour.name,
                },
            )
        return {"domain_flavours": domain_flavours}


@event.listens_for(DomainFlavour.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for domain_flavour in DomainFlavours:
            db_session.add(DomainFlavour(name=domain_flavour.value))
        db_session.commit()
