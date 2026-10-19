import os

import pytest

os.environ.setdefault("MADHUTWIN_BACKEND", "mock")


@pytest.fixture(autouse=True)
def _mock_backend(monkeypatch, request):
    """Unit tests always run on the mock; contract tests opt into the real backend."""
    if request.node.get_closest_marker("contract"):
        monkeypatch.setenv("MADHUTWIN_BACKEND", "real")
        monkeypatch.delenv("MADHUTWIN_DATA", raising=False)
        monkeypatch.delenv("MADHUTWIN_MODEL", raising=False)
    else:
        monkeypatch.setenv("MADHUTWIN_BACKEND", "mock")
        monkeypatch.delenv("MADHUTWIN_DATA", raising=False)
        monkeypatch.delenv("MADHUTWIN_MODEL", raising=False)
    yield
