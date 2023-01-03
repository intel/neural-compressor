# Database migration

Database migration is performed using [Alembic](https://alembic.sqlalchemy.org/en/latest/).

1. Create database on previous version (start INC bench)
2. Update code with latest changes
3. Run command from current directory: `alembic revision --autogenerate -m "Some message"`

Above steps should generate migration file under `alembic/versions` directory.
Database version is checked while INC Bench is starting. When database is outdated migration will be performed automatically.
