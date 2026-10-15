"""Security regression tests for API-key enforcement."""
import pytest
from fastapi import HTTPException

from api import server


def test_development_without_api_key_is_allowed(monkeypatch):
    monkeypatch.setattr(server, "APP_ENV", "development")
    monkeypatch.setattr(server, "API_KEY", "")
    assert server.require_api_key(None) is None


def test_production_without_configured_key_fails_closed(monkeypatch):
    monkeypatch.setattr(server, "APP_ENV", "production")
    monkeypatch.setattr(server, "API_KEY", "")
    with pytest.raises(HTTPException) as exc:
        server.require_api_key(None)
    assert exc.value.status_code == 503


def test_missing_or_invalid_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "APP_ENV", "production")
    monkeypatch.setattr(server, "API_KEY", "test-secret")
    with pytest.raises(HTTPException) as missing:
        server.require_api_key(None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        server.require_api_key("wrong-secret")
    assert invalid.value.status_code == 401


def test_correct_api_key_is_accepted(monkeypatch):
    monkeypatch.setattr(server, "APP_ENV", "production")
    monkeypatch.setattr(server, "API_KEY", "test-secret")
    assert server.require_api_key("test-secret") is None


def test_unknown_history_provider_is_rejected():
    with pytest.raises(HTTPException) as exc:
        server._validate_provider("unexpected")
    assert exc.value.status_code == 422
