"""Chaos tests assert exact event timelines, so they get a clean slate."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
async def clean_execution_state(clean_tasks):
    yield
