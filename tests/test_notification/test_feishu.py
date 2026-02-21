"""Tests for Feishu notifier and signal card formatting (M1.6)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from pangu.models import Action, SignalStatus, TradeSignal
from pangu.notification.feishu import FeishuNotifier, format_signal_card


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


def _llm_signal() -> TradeSignal:
    """Signal with LLM judge metadata."""
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 8, 15),
        symbol="601899",
        name="紫金矿业",
        action=Action.BUY,
        signal_status=SignalStatus.NEW_ENTRY,
        days_in_top_n=0,
        price=18.5,
        confidence=0.78,
        source="llm_judge",
        reason="金价趋势性上涨构成核心驱动，基本面支撑充分",
        factor_score=0.82,
        metadata={
            "bull_reason": "国际金价持续走高，公司矿产金产量稳步增长",
            "bear_reason": "PE_TTM偏高，短期涨幅较大存在回调风险",
            "judge_conclusion": "金价趋势性上涨构成核心驱动，基本面支撑充分",
            "short_term_outlook": "震荡偏强",
            "mid_term_outlook": "看涨",
            "factor_rank": 1,
            "pool_size": 5,
            "factor_details": {
                "rsi_14": 49.12,
                "macd_hist": 0.035,
                "pe_ttm": 22.0,
                "pb": 4.42,
            },
        },
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

    def test_llm_card_contains_factor_rank(self) -> None:
        card = format_signal_card(_llm_signal())
        content = card["elements"][0]["content"]
        assert "排名: 1/5" in content

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

    def test_llm_card_has_collapsible_panel(self) -> None:
        card = format_signal_card(_llm_signal())
        elements = card["elements"]
        # main content + hr + collapsible panel
        assert len(elements) == 3
        assert elements[1]["tag"] == "hr"
        panel = elements[2]
        assert panel["tag"] == "collapsible_panel"
        assert panel["expanded"] is False
        assert "LLM 综合分析" in panel["header"]["title"]["content"]

    def test_llm_card_contains_bull_bear_judge(self) -> None:
        card = format_signal_card(_llm_signal())
        panel_content = card["elements"][2]["elements"][0]["content"]
        assert "牛方观点" in panel_content
        assert "金价持续走高" in panel_content
        assert "熊方观点" in panel_content
        assert "PE_TTM偏高" in panel_content
        assert "裁判结论" in panel_content
        assert "核心驱动" in panel_content

    def test_llm_card_contains_outlook(self) -> None:
        card = format_signal_card(_llm_signal())
        panel_content = card["elements"][2]["elements"][0]["content"]
        assert "短期展望" in panel_content
        assert "震荡偏强" in panel_content
        assert "中期展望" in panel_content
        assert "看涨" in panel_content

    def test_llm_card_contains_factor_details(self) -> None:
        card = format_signal_card(_llm_signal())
        panel_content = card["elements"][2]["elements"][0]["content"]
        assert "关键因子" in panel_content
        assert "RSI(14): 49.1200" in panel_content
        assert "PE(TTM): 22.0000" in panel_content

    def test_no_panel_without_llm_metadata(self) -> None:
        card = format_signal_card(_sell_signal())
        # Only 1 element (main content), no hr/panel
        assert len(card["elements"]) == 1

    def test_llm_card_json_serializable(self) -> None:
        card = format_signal_card(_llm_signal())
        serialized = json.dumps(card, ensure_ascii=False)
        assert "collapsible_panel" in serialized


# ---------------------------------------------------------------------------
# FeishuNotifier tests
# ---------------------------------------------------------------------------


class TestFeishuNotifier:
    def test_send_without_open_id_returns_false(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test")
        result = asyncio.run(notifier.send_signal(_buy_signal()))
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
            result = asyncio.run(notifier.send_signal(_buy_signal()))
        assert result is True

    def test_send_api_failure(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        mock_resp = MagicMock()
        mock_resp.success.return_value = False
        mock_resp.code = 99999
        mock_resp.msg = "test error"
        with patch.object(notifier._client.im.v1.message, "create", return_value=mock_resp):
            result = asyncio.run(notifier.send_signal(_buy_signal()))
        assert result is False

    def test_send_exception(self) -> None:
        notifier = FeishuNotifier(app_id="test", app_secret="test", open_id="ou_123")
        with patch.object(
            notifier._client.im.v1.message, "create", side_effect=RuntimeError("network")
        ):
            result = asyncio.run(notifier.send_signal(_buy_signal()))
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
