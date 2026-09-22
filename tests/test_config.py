"""Configuration validation is pure and never reaches network or disk data."""

from pathlib import Path

import pytest

from src import config


def test_data_plane_is_strict_and_normalized():
    assert config.get_data_plane(" DEMO ") == "demo"
    assert config.get_data_plane("research") == "research"
    with pytest.raises(config.ConfigurationError, match="expected one of"):
        config.get_data_plane("production")


@pytest.mark.parametrize("url", [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://127.42.0.9:11434",
    "http://[::1]:11434",
    "http://host.docker.internal:11434",
])
def test_research_model_endpoint_allows_explicitly_local_hosts(url):
    assert config.validate_model_endpoint(url, plane="research", allow_remote=False) == url


def test_research_model_endpoint_refuses_remote_without_explicit_override():
    url = "https://models.example.com/ollama/"
    with pytest.raises(config.ConfigurationError, match="refuses remote"):
        config.validate_model_endpoint(url, plane="research", allow_remote=False)
    assert config.validate_model_endpoint(url, plane="research", allow_remote=True) == url[:-1]
    assert config.validate_model_endpoint(url, plane="demo", allow_remote=False) == url[:-1]


@pytest.mark.parametrize("url", ["models.example.com", "ftp://localhost/model", "http:///missing"])
def test_model_endpoint_requires_http_url_with_host(url):
    with pytest.raises(config.ConfigurationError, match="invalid model endpoint"):
        config.validate_model_endpoint(url, plane="demo")


def test_default_paths_are_repository_relative():
    assert config.REPO_ROOT == Path(__file__).resolve().parents[1]
    assert config.DATA_DIR == config.REPO_ROOT / "data"
    assert config.MODELS_DIR == config.REPO_ROOT / "models"
