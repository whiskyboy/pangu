"""Feishu (Lark) notifier — PRD §4.5.2.

Uses lark-oapi SDK for:
- Auto-pairing: user sends any DM to bot → open_id persisted via WS long connection
- Signal push: interactive card via OpenAPI im.v1.message.create
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Protocol

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)

from trading_agent.models import Action, SignalStatus, TradeSignal

logger = logging.getLogger(__name__)


class NotificationProvider(Protocol):
    """Interface for sending trade signal notifications."""

    async def send(self, signal: TradeSignal) -> bool:
        """Send a trade signal notification. Return True on success."""
        ...


# ---------------------------------------------------------------------------
# Feishu notifier
# ---------------------------------------------------------------------------

_ACTION_ICON = {
    Action.BUY: "🟢 买入信号",
    Action.SELL: "🔴 卖出信号",
    Action.HOLD: "⚪ 持有观望",
}

_STATUS_LABEL = {
    SignalStatus.NEW_ENTRY: "⚡ 首次入选",
    SignalStatus.SUSTAINED: "🔄 维持关注",
    SignalStatus.EXIT: "⛔ 退出",
}


def _stars(confidence: float) -> str:
    """Convert 0-1 confidence to star rating."""
    n = max(1, min(5, round(confidence * 5)))
    return "⭐" * n


def format_signal_card(signal: TradeSignal) -> dict:
    """Build a Feishu interactive card JSON for *signal*."""
    ts = signal.timestamp.strftime("%Y-%m-%d %H:%M")
    header = f"{_ACTION_ICON.get(signal.action, '⚪')} | {ts}"
    status = _STATUS_LABEL.get(signal.signal_status, str(signal.signal_status.value))

    lines = [
        f"**股票**: {signal.name} ({signal.symbol})",
        f"**信号状态**: {status}",
    ]

    if signal.factor_score is not None:
        lines.append(f"**因子评分**: {signal.factor_score:.2f}")

    action_verb = "买入" if signal.action is Action.BUY else "卖出" if signal.action is Action.SELL else "观望"
    lines.append(f"**建议操作**: 以 ¥{signal.price:,.2f} {action_verb}")

    if signal.stop_loss is not None:
        pct = (signal.stop_loss - signal.price) / signal.price * 100
        lines.append(f"**止损价**: ¥{signal.stop_loss:,.2f} ({pct:+.1f}%)")
    if signal.take_profit is not None:
        pct = (signal.take_profit - signal.price) / signal.price * 100
        lines.append(f"**止盈价**: ¥{signal.take_profit:,.2f} ({pct:+.1f}%)")

    lines.append(f"**置信度**: {_stars(signal.confidence)} ({signal.confidence:.2f})")

    if signal.reason:
        lines.append(f"\n📊 **分析**: {signal.reason}")

    # Metadata extras
    meta = signal.metadata or {}
    if "news_title" in meta:
        lines.append(f"📰 **事件**: {meta['news_title']}")

    lines.append(f"\n_信号来源: {signal.source}_")

    content = "\n".join(lines)

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": header},
            "template": (
                "green" if signal.action is Action.BUY
                else "red" if signal.action is Action.SELL
                else "grey"
            ),
        },
        "elements": [
            {"tag": "markdown", "content": content},
        ],
    }


class FeishuNotifier:
    """Send trade signals via Feishu bot private message.

    Supports auto-pairing: call ``start_ws_client()`` to listen for user DMs.
    When a user sends any message to the bot, their ``open_id`` is captured
    and stored via the ``on_bind`` callback.
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        open_id: str | None = None,
        on_bind: Callable[[str], None] | None = None,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .build()
        )
        self._open_id = open_id
        self._on_bind = on_bind
        self._ws_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()

    @property
    def open_id(self) -> str | None:
        return self._open_id

    @open_id.setter
    def open_id(self, value: str) -> None:
        self._open_id = value

    # -----------------------------------------------------------------
    # WS long connection for auto-pairing
    # -----------------------------------------------------------------

    def _handle_message(self, event: P2ImMessageReceiveV1) -> None:
        """Handle incoming DM: extract open_id and reply with confirmation."""
        try:
            open_id = event.event.sender.sender_id.open_id
        except AttributeError:
            logger.warning("Could not extract open_id from event")
            return

        if not open_id:
            return

        self._open_id = open_id
        logger.info("Feishu pairing: bound open_id=%s", open_id)

        if self._on_bind:
            try:
                self._on_bind(open_id)
            except Exception:
                logger.exception("on_bind callback failed for open_id=%s", open_id)

        # Reply with confirmation
        self._reply_text(open_id, "✅ 绑定成功，后续交易信号将私聊推送给你")

    def _reply_text(self, open_id: str, text: str) -> None:
        """Send a plain-text reply to the user."""
        content = json.dumps({"text": text})
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(open_id)
                .msg_type("text")
                .content(content)
                .build()
            )
            .build()
        )
        try:
            with self._send_lock:
                resp = self._client.im.v1.message.create(request)
            if not resp.success():
                logger.error("Pairing reply failed: code=%s msg=%s", resp.code, resp.msg)
        except Exception:
            logger.exception("Pairing reply exception")

    def start_ws_client(self) -> None:
        """Start the WebSocket long connection in a background daemon thread.

        The WS client listens for ``im.message.receive_v1`` events.
        When a user sends any DM to the bot, ``_handle_message`` is called.
        This method is non-blocking — it starts a daemon thread and returns immediately.
        """
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message)
            .build()
        )

        ws_client = lark.ws.Client(
            app_id=self._app_id,
            app_secret=self._app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.WARNING,
        )

        self._ws_thread = threading.Thread(
            target=ws_client.start,
            name="feishu-ws",
            daemon=True,
        )
        self._ws_thread.start()
        logger.info("Feishu WS client started (daemon thread)")

    @property
    def ws_running(self) -> bool:
        """Whether the WS background thread is alive."""
        return self._ws_thread is not None and self._ws_thread.is_alive()

    # -----------------------------------------------------------------
    # Push signal via OpenAPI
    # -----------------------------------------------------------------

    async def send(self, signal: TradeSignal) -> bool:
        if not self._open_id:
            logger.warning("FeishuNotifier: no open_id bound, skipping push")
            return False

        card = format_signal_card(signal)
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(self._open_id)
                .msg_type("interactive")
                .content(json.dumps(card))
                .build()
            )
            .build()
        )

        try:
            with self._send_lock:
                resp = self._client.im.v1.message.create(request)
            if resp.success():
                logger.info("Feishu message sent to %s", self._open_id)
                return True
            logger.error("Feishu send failed: code=%s msg=%s", resp.code, resp.msg)
            return False
        except Exception:
            logger.exception("Feishu send exception")
            return False
