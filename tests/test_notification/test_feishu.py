"""Tests for FeishuNotifier (pairing + markdown/text push)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from pangu.notification.feishu import FeishuNotifier

# ---------------------------------------------------------------------------
# Notifier basics
# ---------------------------------------------------------------------------


class TestFeishuNotifier:
    def test_open_id_property(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        assert notifier.open_id == "ou_123"
        notifier.open_id = "ou_456"
        assert notifier.open_id == "ou_456"

    def test_send_text_without_open_id_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        result = asyncio.run(notifier.send_text("hello"))
        assert result is False

    def test_send_markdown_without_open_id_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        result = asyncio.run(notifier.send_markdown("title", "**body**"))
        assert result is False

    def test_send_text_success(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            assert asyncio.run(notifier.send_text("hi")) is True

    def test_send_markdown_success(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp) as mock_create:
            assert asyncio.run(notifier.send_markdown("title", "body")) is True
        # Verify it sent an interactive card
        req = mock_create.call_args[0][0]
        assert req.body.msg_type == "interactive"

    def test_send_api_failure_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = False
        mock_resp.code = 99999
        mock_resp.msg = "test error"
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            assert asyncio.run(notifier.send_text("hi")) is False

    def test_send_exception_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        with patch.object(
            notifier._client.im.v1.message,
            "create",
            side_effect=RuntimeError("network"),
        ):
            assert asyncio.run(notifier.send_text("hi")) is False


# ---------------------------------------------------------------------------
# Pairing — handle_message callback flow
# ---------------------------------------------------------------------------


def _make_event(open_id: str = "ou_abc123") -> MagicMock:
    event = MagicMock()
    event.event.sender.sender_id.open_id = open_id
    return event


class TestPairing:
    def test_handle_message_sets_open_id(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        assert notifier.open_id is None

        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            notifier._handle_message(_make_event("ou_new_user"))
        assert notifier.open_id == "ou_new_user"

    def test_handle_message_calls_on_bind(self) -> None:
        bound_ids: list[str] = []
        notifier = FeishuNotifier(
            app_id="test",
            app_secret="test",
            on_bind=lambda oid: bound_ids.append(oid),
        )
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            notifier._handle_message(_make_event("ou_callback"))
        assert bound_ids == ["ou_callback"]

    def test_handle_message_sends_reply(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(
            notifier._client.im.v1.message,
            "create",
            return_value=mock_resp,
        ) as mock_create:
            notifier._handle_message(_make_event("ou_reply"))
        mock_create.assert_called_once()
        req = mock_create.call_args[0][0]
        assert req.body.msg_type == "text"
        content = json.loads(req.body.content)
        assert "绑定成功" in content["text"]

    def test_handle_message_bad_event_does_not_raise(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        bad_event = MagicMock()
        bad_event.event = None
        notifier._handle_message(bad_event)  # should not raise
        assert notifier.open_id is None

    def test_on_bind_exception_does_not_crash(self) -> None:
        def bad_callback(oid: str) -> None:
            raise RuntimeError("db down")

        notifier = FeishuNotifier(
            app_id="test",
            app_secret="test",
            on_bind=bad_callback,
        )
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            notifier._handle_message(_make_event("ou_safe"))
        # open_id is still set even if callback fails
        assert notifier.open_id == "ou_safe"
