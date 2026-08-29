"""Alembic environment.

Deliberately synchronous: `postgresql+psycopg` is the same dialect the app
uses asynchronously, so there is one URL and one driver, but migrations have
no reason to be async and a sync engine keeps this file boring.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from acp.config import settings
from acp.db.models import metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
