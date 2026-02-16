"""Tests for the config loader (M1.3)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trading_agent.config import get_settings, load_settings, reset_settings


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
            auto_filter_st = true

            [strategy]
            buy_threshold = 0.7
        """)
        s = load_settings(path)
        assert s.system["timezone"] == "Asia/Shanghai"
        assert s.stock_pool["auto_filter_st"] is True
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
            [notification.email]
            smtp_host = "$NONEXISTENT_VAR"
        """)
        s = load_settings(path)
        assert s.notification["email"]["smtp_host"] == ""

    def test_list_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMAIL1", "a@b.com")
        path = _write_toml(tmp_path, """
            [notification.email]
            to_addresses = ["$EMAIL1"]
        """)
        s = load_settings(path)
        assert s.notification["email"]["to_addresses"] == ["a@b.com"]

    def test_non_string_values_unchanged(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [strategy]
            buy_threshold = 0.7
            max_signals_per_day_per_stock_per_direction = 1
        """)
        s = load_settings(path)
        assert s.strategy["buy_threshold"] == 0.7
        assert s.strategy["max_signals_per_day_per_stock_per_direction"] == 1


# ---------------------------------------------------------------------------
# LLM fallback auto-pruning
# ---------------------------------------------------------------------------

class TestFallbackPruning:
    def test_prunes_provider_without_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        path = _write_toml(tmp_path, """
            [llm]
            fallback_providers = ["deepseek/deepseek-chat", "gemini/gemini-2.5-flash"]
        """)
        s = load_settings(path)
        assert s.llm["fallback_providers"] == []

    def test_keeps_provider_with_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-123")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        path = _write_toml(tmp_path, """
            [llm]
            fallback_providers = ["deepseek/deepseek-chat", "gemini/gemini-2.5-flash"]
        """)
        s = load_settings(path)
        assert s.llm["fallback_providers"] == ["deepseek/deepseek-chat"]

    def test_keeps_all_when_all_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-123")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-456")
        path = _write_toml(tmp_path, """
            [llm]
            fallback_providers = ["deepseek/deepseek-chat", "gemini/gemini-2.5-flash"]
        """)
        s = load_settings(path)
        assert len(s.llm["fallback_providers"]) == 2

    def test_unknown_provider_kept(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [llm]
            fallback_providers = ["custom/my-model"]
        """)
        s = load_settings(path)
        assert s.llm["fallback_providers"] == ["custom/my-model"]

    def test_whitespace_only_key_treated_as_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "  ")
        path = _write_toml(tmp_path, """
            [llm]
            fallback_providers = ["deepseek/deepseek-chat"]
        """)
        s = load_settings(path)
        assert s.llm["fallback_providers"] == []


# ---------------------------------------------------------------------------
# Notification channel auto-detection
# ---------------------------------------------------------------------------

class TestChannelAutoDetection:
    def test_prunes_channel_without_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        monkeypatch.setenv("FEISHU_APP_ID", "id")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
        path = _write_toml(tmp_path, """
            [notification]
            channels = ["feishu", "email"]
        """)
        s = load_settings(path)
        assert s.notification["channels"] == ["feishu"]

    def test_keeps_both_when_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FEISHU_APP_ID", "id")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_USER", "user")
        monkeypatch.setenv("SMTP_PASSWORD", "pass")
        path = _write_toml(tmp_path, """
            [notification]
            channels = ["feishu", "email"]
        """)
        s = load_settings(path)
        assert s.notification["channels"] == ["feishu", "email"]

    def test_prunes_all_unconfigured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FEISHU_APP_ID", raising=False)
        monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        path = _write_toml(tmp_path, """
            [notification]
            channels = ["feishu", "email"]
        """)
        s = load_settings(path)
        assert s.notification["channels"] == []

    def test_partial_smtp_config_prunes_email(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        path = _write_toml(tmp_path, """
            [notification]
            channels = ["email"]
        """)
        s = load_settings(path)
        assert s.notification["channels"] == []

    def test_unknown_channel_kept(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [notification]
            channels = ["slack"]
        """)
        s = load_settings(path)
        assert s.notification["channels"] == ["slack"]


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
