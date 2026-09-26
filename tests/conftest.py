from __future__ import annotations

import pytest

from triage import documents


@pytest.fixture(autouse=True)
def _clear_response_cache() -> None:
    """The classifier response cache is process-wide; isolate it per test."""

    documents._RESPONSE_CACHE.clear()
