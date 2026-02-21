"""Tests for the config loader (M1.3)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pangu.config import get_settings, load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset the singleton before every test."""
    reset_settings()
    yield  # type: ignore[misc]
    reset_settings()


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "settings.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

class TestBasicLoading:
    def test_loads_sections(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [system]
            timezone = "Asia/Shanghai"
            log_level = "INFO"

            [stock_pool]
            min_listing_days = 60

            [strategy]
            buy_threshold = 0.7
        """)
        s = load_settings(path)
        assert s.system["timezone"] == "Asia/Shanghai"
        assert s.stock_pool["min_listing_days"] == 60
        assert s.strategy["buy_threshold"] == 0.7

    def test_missing_section_defaults_to_empty(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [system]
            timezone = "UTC"
        """)
        s = load_settings(path)
        assert s.llm == {}
        assert s.notification == {}


# ---------------------------------------------------------------------------
# Env-var substitution
# ---------------------------------------------------------------------------

class TestEnvVarSubstitution:
    def test_string_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_SECRET", "hunter2")
        path = _write_toml(tmp_path, """
            [notification.feishu]
            app_id = "$MY_SECRET"
        """)
        s = load_settings(path)
        assert s.notification["feishu"]["app_id"] == "hunter2"

    def test_inline_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_DEPLOYMENT", "gpt-5.2")
        path = _write_toml(tmp_path, """
            [llm]
            provider = "azure/$AZURE_DEPLOYMENT"
        """)
        s = load_settings(path)
        assert s.llm["provider"] == "azure/gpt-5.2"

    def test_missing_env_var_resolves_to_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        path = _write_toml(tmp_path, """
            [notification.feishu]
            app_id = "$NONEXISTENT_VAR"
        """)
        s = load_settings(path)
        assert s.notification["feishu"]["app_id"] == ""

    def test_list_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_MODEL", "my-model")
        path = _write_toml(tmp_path, """
            [llm]
            extra = ["custom/$MY_MODEL"]
        """)
        s = load_settings(path)
        assert s.llm["extra"] == ["custom/my-model"]

    def test_non_string_values_unchanged(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [strategy]
            buy_threshold = 0.7
            top_n = 10
        """)
        s = load_settings(path)
        assert s.strategy["buy_threshold"] == 0.7
        assert s.strategy["top_n"] == 10


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_settings_returns_same_instance(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [system]
            timezone = "UTC"
        """)
        s1 = get_settings(path)
        s2 = get_settings(path)
        assert s1 is s2

    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [system]
            timezone = "UTC"
        """)
        s1 = get_settings(path)
        reset_settings()
        s2 = get_settings(path)
        assert s1 is not s2
