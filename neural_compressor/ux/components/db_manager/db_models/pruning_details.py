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
"""The PruningDetails class."""
import json
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import session

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.utils.workload.pruning import Pruning as PruningConfig


class PruningDetails(Base):
    """INC Bench pruning details' table representation."""

    __tablename__ = "pruning_details"

    id = Column(Integer, primary_key=True, index=True, unique=True)
    train = Column(String)
    approach = Column(String)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    modified_at = Column(DateTime, nullable=True, onupdate=func.now())

    @staticmethod
    def add(
        db_session: session.Session,
        pruning_details: PruningConfig,
    ) -> int:
        """
        Add pruning details to database.

        returns id of added pruning details
        """
        train: Optional[dict] = None
        approach: Optional[dict] = None

        if pruning_details.train is not None:
            train = pruning_details.train.serialize()  # type: ignore
        if pruning_details.approach is not None:
            approach = pruning_details.approach.serialize()  # type: ignore

        new_pruning_details = PruningDetails(
            train=json.dumps(train),
            approach=json.dumps(approach),
        )
        db_session.add(new_pruning_details)
        db_session.flush()

        return int(new_pruning_details.id)

    @staticmethod
    def update(
        db_session: session.Session,
        pruning_details_id: int,
        pruning_details_data: PruningConfig,
    ) -> dict:
        """Update pruning details."""
        pruning_details = (
            db_session.query(PruningDetails).filter(PruningDetails.id == pruning_details_id).one()
        )

        train = None
        approach = None
        if pruning_details_data.train is not None:
            train = json.dumps(pruning_details_data.train.serialize())

        if pruning_details_data.approach is not None:
            approach = json.dumps(pruning_details_data.approach.serialize())
        pruning_details.train = train
        pruning_details.approach = approach

        db_session.add(pruning_details)
        db_session.flush()

        updated_train = None
        if pruning_details.train is not None:
            updated_train = json.loads(pruning_details.train)

        updated_approach = None
        if pruning_details.approach is not None:
            updated_approach = json.loads(pruning_details.approach)

        return {
            "id": pruning_details.id,
            "train": updated_train,
            "approach": updated_approach,
            "created_at": str(pruning_details.created_at),
            "modified_at": str(pruning_details.modified_at),
        }

    @staticmethod
    def delete_pruning_details(
        db_session: session.Session,
        pruning_details_id: int,
    ) -> Optional[int]:
        """Remove pruning_details from database."""
        pruning_details = (
            db_session.query(PruningDetails)
            .filter(PruningDetails.id == pruning_details_id)
            .one_or_none()
        )
        if pruning_details is None:
            return None
        db_session.delete(pruning_details)
        db_session.flush()

        return int(pruning_details.id)

    @staticmethod
    def build_info(
        pruning_details: Any,
    ) -> dict:
        """Build profiling result info."""
        return {
            "id": pruning_details.id,
            "train": json.loads(pruning_details.train),
            "approach": json.loads(pruning_details.approach),
            "created_at": str(pruning_details.created_at),
            "modified_at": str(pruning_details.modified_at),
        }
