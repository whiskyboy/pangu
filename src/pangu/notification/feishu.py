"""Feishu (Lark) notifier.

Uses lark-oapi SDK for:
- Auto-pairing: user sends any DM to bot → open_id persisted via WS long connection
- Markdown card push via OpenAPI im.v1.message.create
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)

from pangu.notification import NotificationProvider  # noqa: F401 — backward compat

logger = logging.getLogger(__name__)


class FeishuNotifier:
    """Send notifications via Feishu bot private message.

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
        self._client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
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
                CreateMessageRequestBody.builder().receive_id(open_id).msg_type("text").content(content).build()
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

    # -----------------------------------------------------------------
    # Push via OpenAPI
    # -----------------------------------------------------------------

    def _send_message(self, msg_type: str, content: str) -> bool:
        """Send a message to the bound user. Returns True on success."""
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder().receive_id(self._open_id).msg_type(msg_type).content(content).build()
            )
            .build()
        )
        try:
            with self._send_lock:
                resp = self._client.im.v1.message.create(request)
            if resp.success():
                logger.info("Feishu %s sent to %s", msg_type, self._open_id)
                return True
            logger.error("Feishu send failed: code=%s msg=%s", resp.code, resp.msg)
            return False
        except Exception:
            logger.exception("Feishu send exception")
            return False

    async def send_text(self, text: str) -> bool:
        """Send a plain-text message."""
        if not self._open_id:
            logger.warning("FeishuNotifier: no open_id bound, skipping push")
            return False
        return self._send_message("text", json.dumps({"text": text}))

    async def send_markdown(self, title: str, content: str) -> bool:
        """Send a markdown card."""
        if not self._open_id:
            logger.warning("FeishuNotifier: no open_id bound, skipping push")
            return False
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": content}],
        }
        return self._send_message("interactive", json.dumps(card))
