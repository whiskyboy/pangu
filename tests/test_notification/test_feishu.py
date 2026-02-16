"""Tests for Feishu notifier and signal card formatting (M1.6)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from trading_agent.models import Action, SignalStatus, TradeSignal
from trading_agent.notification.feishu import FeishuNotifier, format_signal_card


def _buy_signal() -> TradeSignal:
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 10, 32),
        symbol="600519",
        name="贵州茅台",
        action=Action.BUY,
        signal_status=SignalStatus.NEW_ENTRY,
        days_in_top_n=0,
        price=1850.0,
        confidence=0.82,
        source="merged",
        reason="multi-factor top-1",
        stop_loss=1795.5,
        take_profit=1961.0,
        factor_score=0.85,
        metadata={"news_title": "茅台一季度营收同比增长18%"},
    )


def _sell_signal() -> TradeSignal:
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 15, 0),
        symbol="601318",
        name="中国平安",
        action=Action.SELL,
        signal_status=SignalStatus.EXIT,
        days_in_top_n=5,
        price=48.0,
        confidence=0.65,
        source="factor",
        reason="factor exit",
    )


# ---------------------------------------------------------------------------
# Card formatting tests
# ---------------------------------------------------------------------------


class TestFormatSignalCard:
    def test_buy_card_header(self) -> None:
        card = format_signal_card(_buy_signal())
        assert card["header"]["template"] == "green"
        assert "买入" in card["header"]["title"]["content"]

    def test_sell_card_header(self) -> None:
        card = format_signal_card(_sell_signal())
        assert card["header"]["template"] == "red"
        assert "卖出" in card["header"]["title"]["content"]

    def test_card_contains_stock_info(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "贵州茅台" in content
        assert "600519" in content

    def test_card_contains_price(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "1,850.00" in content
        assert "买入" in content

    def test_card_contains_stop_loss_take_profit(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "止损" in content
        assert "止盈" in content
        assert "1,795.50" in content
        assert "1,961.00" in content

    def test_card_contains_confidence_stars(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "⭐" in content
        assert "0.82" in content

    def test_card_contains_factor_score(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "0.85" in content

    def test_card_contains_signal_status(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "首次入选" in content

    def test_card_contains_news_title(self) -> None:
        card = format_signal_card(_buy_signal())
        content = card["elements"][0]["content"]
        assert "茅台一季度营收同比增长18%" in content

    def test_card_without_optional_fields(self) -> None:
        card = format_signal_card(_sell_signal())
        content = card["elements"][0]["content"]
        assert "止损" not in content
        assert "止盈" not in content

    def test_card_json_serializable(self) -> None:
        card = format_signal_card(_buy_signal())
        serialized = json.dumps(card, ensure_ascii=False)
        assert isinstance(serialized, str)

    def test_hold_card_template(self) -> None:
        sig = TradeSignal(
            timestamp=datetime(2026, 1, 1),
            symbol="000001", name="平安银行",
            action=Action.HOLD, signal_status=SignalStatus.SUSTAINED,
            days_in_top_n=3, price=12.0, confidence=0.5,
            source="merged", reason="conflicting signals",
        )
        card = format_signal_card(sig)
        assert card["header"]["template"] == "grey"


# ---------------------------------------------------------------------------
# FeishuNotifier tests
# ---------------------------------------------------------------------------


class TestFeishuNotifier:
    def test_send_without_open_id_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        result = asyncio.run(notifier.send(_buy_signal()))
        assert result is False

    def test_open_id_property(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        assert notifier.open_id == "ou_123"
        notifier.open_id = "ou_456"
        assert notifier.open_id == "ou_456"

    def test_send_success(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            result = asyncio.run(notifier.send(_buy_signal()))
        assert result is True

    def test_send_api_failure(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = False
        mock_resp.code = 99999
        mock_resp.msg = "test error"
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            result = asyncio.run(notifier.send(_buy_signal()))
        assert result is False

    def test_send_exception(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        with patch.object(
            notifier._client.im.v1.message, "create", side_effect=RuntimeError("network")
        ):
            result = asyncio.run(notifier.send(_buy_signal()))
        assert result is False


# ---------------------------------------------------------------------------
# WS pairing tests
# ---------------------------------------------------------------------------


def _make_event(open_id: str = "ou_abc123") -> MagicMock:
    """Build a mock P2ImMessageReceiveV1 event."""
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
            app_id="test", app_secret="test",
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
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp) as mock_create:
            notifier._handle_message(_make_event("ou_reply"))

        mock_create.assert_called_once()
        # Verify the reply was a text message containing confirmation
        req = mock_create.call_args[0][0]
        assert req.body.msg_type == "text"
        content = json.loads(req.body.content)
        assert "绑定成功" in content["text"]

    def test_handle_message_bad_event(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        bad_event = MagicMock()
        bad_event.event = None
        # Should not raise
        notifier._handle_message(bad_event)
        assert notifier.open_id is None

    def test_on_bind_exception_does_not_crash(self) -> None:
        def bad_callback(oid: str) -> None:
            raise RuntimeError("db down")

        notifier = FeishuNotifier(
            app_id="test", app_secret="test", on_bind=bad_callback,
        )
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            # Should not raise despite bad callback
            notifier._handle_message(_make_event("ou_safe"))
        # open_id should still be set even if callback fails
        assert notifier.open_id == "ou_safe"

    def test_ws_not_running_initially(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        assert notifier.ws_running is False

    def test_start_ws_client_spawns_thread(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        with patch("lark_oapi.ws.Client") as mock_ws_cls:
            mock_ws_instance = MagicMock()
            # Make start() block briefly then return
            mock_ws_instance.start.return_value = None
            mock_ws_cls.return_value = mock_ws_instance

            notifier.start_ws_client()

            # Thread was started
            assert notifier._ws_thread is not None
            assert notifier._ws_thread.daemon is True
            notifier._ws_thread.join(timeout=2)

            mock_ws_cls.assert_called_once()
            mock_ws_instance.start.assert_called_once()
