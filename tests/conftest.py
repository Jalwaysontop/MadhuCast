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


@pytest.fixture(scope="session")
def p3_exports():
    """P3 pipeline -> contract files, built once (demo cohort unless data/raw/cgmacros exists)."""
    from src.serving import p3_export

    return p3_export.build()


@pytest.fixture
def real_data(p3_exports, tmp_path, monkeypatch):
    """Point MADHUTWIN_DATA=real at a fresh export in tmp_path."""
    from src.serving import backend, p3_export

    exports, _ = p3_exports
    p3_export.write(exports, tmp_path)
    monkeypatch.setattr(backend, "PROFILES_DIR", tmp_path / "patient_profiles")
    monkeypatch.setattr(backend, "TIMESERIES_DIR", tmp_path / "timeseries")
    monkeypatch.setattr(backend, "MEALS_DIR", tmp_path / "meals")
    monkeypatch.setenv("MADHUTWIN_DATA", "real")
    backend.clear_caches()
    yield exports
    backend.clear_caches()
