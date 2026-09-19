"""Tracing egress policy + degradation. No network: the Langfuse client is faked."""

import json
import importlib
from contextlib import contextmanager

import pytest
import langfuse

import src.obs.tracing as tracing

SECRET = "sk-lf-test-secret-do-not-log"


class FakeClient:
    def __init__(self, auth=True, fail_span=False):
        self.auth, self.fail_span, self.flushed, self.spans = auth, fail_span, 0, []

    def auth_check(self):
        return self.auth

    @contextmanager
    def start_as_current_observation(self, as_type, name, **kw):
        if self.fail_span:
            raise RuntimeError("sdk broke")
        self.spans.append(name)
        yield type("S", (), {"update": lambda *a, **k: None})()

    def update_current_span(self, **kw):
        pass

    def flush(self):
        self.flushed += 1


@pytest.fixture
def configure(monkeypatch):
    def _cfg(plane="demo", enabled="1", url=None, keys=True, fake=None):
        monkeypatch.setenv("LUMEN_DATA_PLANE", plane)
        monkeypatch.setenv("LUMEN_TRACING", enabled)
        for k in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
            monkeypatch.delenv(k, raising=False)
        if url:
            monkeypatch.setenv("LANGFUSE_BASE_URL", url)
        if keys:
            monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
            monkeypatch.setenv("LANGFUSE_SECRET_KEY", SECRET)
        else:
            monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
            monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        calls = []
        monkeypatch.setattr(langfuse, "get_client", lambda: calls.append(1) or (fake or FakeClient()))
        importlib.reload(tracing)
        return calls
    yield _cfg
    monkeypatch.setenv("LUMEN_TRACING", "0")
    importlib.reload(tracing)


def test_off_by_default(configure):
    calls = configure(enabled="0", url="https://cloud.example.com")
    assert tracing.client() is None and tracing.status()["state"] == "off" and not calls


@pytest.mark.parametrize("url", ["https://cloud.langfuse.com", None])   # None -> SDK default = cloud
def test_research_plane_refuses_remote(configure, url):
    calls = configure(plane="research", url=url)
    assert tracing.client() is None and not calls
    assert tracing.status()["state"] == "refused"


def test_research_plane_allows_local(configure):
    configure(plane="research", url="http://localhost:3000")
    assert tracing.client() is not None and tracing.status()["state"] == "active"


def test_demo_plane_remote_https_allowed_and_status_has_no_secrets(configure):
    configure(plane="demo", url="https://cloud.langfuse.com")
    assert tracing.client() is not None
    st = tracing.status()
    assert st == {"enabled": True, "provider": "langfuse", "host": "cloud.langfuse.com", "state": "active",
                  "policy": "remote endpoint (demo plane)", "keys_configured": True}
    assert SECRET not in json.dumps(st) and "pk-lf-test" not in json.dumps(st)


def test_demo_plane_refuses_plain_http_remote(configure):
    configure(plane="demo", url="http://traces.example.com")
    assert tracing.client() is None and tracing.status()["state"] == "refused"


def test_missing_keys_and_failed_auth_degrade(configure):
    configure(plane="demo", url="https://cloud.langfuse.com", keys=False)
    assert tracing.client() is None and tracing.status()["state"] == "misconfigured"
    configure(plane="demo", url="https://cloud.langfuse.com", fake=FakeClient(auth=False))
    assert tracing.client() is None and tracing.status()["state"] == "unavailable"


def test_root_trace_degrades_but_app_errors_propagate(configure):
    configure(plane="demo", url="https://cloud.langfuse.com", fake=FakeClient(fail_span=True))
    with tracing.root_trace("lumen.graph", session_id="t", metadata={"request_id": "r"}, tags=["x"]) as s:
        assert s is None                                   # SDK failure -> no-op, request continues
    with pytest.raises(ValueError):
        with tracing.root_trace("lumen.graph", session_id="t", metadata={}, tags=[]):
            raise ValueError("application error")


def test_root_trace_nests_and_flushes(configure):
    fake = FakeClient()
    configure(plane="demo", url="https://cloud.langfuse.com", fake=fake)
    with tracing.root_trace("lumen.graph", session_id="api-r1", metadata={"request_id": "r1"}, tags=["api"]):
        with tracing.generation("ollama:main", "qwen3:8b", prompt=[]):
            pass
    tracing.flush()
    assert fake.spans == ["lumen.graph", "ollama:main"] and fake.flushed == 1
