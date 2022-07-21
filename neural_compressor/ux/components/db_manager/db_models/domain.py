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
"""Domain package contains domain class representing domain table in database."""
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, event
from sqlalchemy.orm import relationship, session, sessionmaker
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base, DBManager
from neural_compressor.ux.components.db_manager.db_models.transform import (
    transform_domain_association,
)
from neural_compressor.ux.utils.consts import Domains
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Domain table")

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class Domain(Base):
    """INC Bench Domain table class."""

    __tablename__ = "domain"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    models: Any = relationship("Model", back_populates="domain")
    transforms: Any = relationship(
        "Transform",
        secondary=transform_domain_association,
        back_populates="domains",
    )

    @staticmethod
    def get_domain_id(db_session: session.Session, domain_name: str) -> int:
        """Get domain id from name."""
        domain_ids = db_session.query(Domain.id).filter(Domain.name.ilike(domain_name))
        log.debug("Found domains:")
        for domain_id in domain_ids:
            log.debug(domain_id)
        found_domains = domain_ids.count()
        if found_domains < 1:
            raise Exception(f"Domain {domain_name} is not supported.")

        if found_domains > 1:
            raise Exception("Domain name is not unique.")
        return int(domain_ids[0].id)

    @staticmethod
    def list(db_session: session.Session) -> dict:
        """List available domains."""
        domains = []
        domain_instances = db_session.query(
            Domain.id,
            Domain.name,
        ).order_by(Domain.id)
        for domain in domain_instances:
            domains.append(
                {
                    "id": domain.id,
                    "name": domain.name,
                },
            )
        return {"domains": domains}


@event.listens_for(Domain.__table__, "after_create")
def fill_dictionary(*args: list, **kwargs: dict) -> None:
    """Fill dictionary with default values."""
    with Session.begin() as db_session:
        for domain in Domains:
            db_session.add(Domain(name=domain.value))
