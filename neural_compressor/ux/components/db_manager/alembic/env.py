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
# flake8: noqa
# mypy: ignore-errors

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from neural_compressor.ux.components.db_manager.db_manager import Base

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.benchmark_result import BenchmarkResult
from neural_compressor.ux.components.db_manager.db_models.dataloader import Dataloader
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.domain import Domain
from neural_compressor.ux.components.db_manager.db_models.domain_flavour import DomainFlavour
from neural_compressor.ux.components.db_manager.db_models.example import Example
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.components.db_manager.db_models.metric import Metric
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.profiling import Profiling
from neural_compressor.ux.components.db_manager.db_models.profiling_result import ProfilingResult
from neural_compressor.ux.components.db_manager.db_models.project import Project
from neural_compressor.ux.components.db_manager.db_models.transform import Transform
from neural_compressor.ux.components.db_manager.db_models.tuning_details import TuningDetails
from neural_compressor.ux.components.db_manager.db_models.tuning_history import TuningHistory

db_manager = DBManager()


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = db_manager.database_entrypoint
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    config_ini = config.get_section(config.config_ini_section)
    config_ini["sqlalchemy.url"] = db_manager.database_entrypoint
    connectable = engine_from_config(
        config_ini,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
