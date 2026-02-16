#!/usr/bin/env python3
"""Feishu pairing utility — obtain your open_id by sending a DM to the bot.

Usage:
    export FEISHU_APP_ID=cli_xxx
    export FEISHU_APP_SECRET=xxx
    python scripts/feishu_pairing.py

Then open Feishu and send any message to the bot. This script will:
1. Print your open_id
2. Reply with a confirmation message
3. Exit automatically

Copy the open_id into your .env file as FEISHU_OPEN_ID=<value>.
"""

from __future__ import annotations

import json
import os
import sys
import threading

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)


def main() -> None:
    app_id = os.environ.get("FEISHU_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        print("ERROR: FEISHU_APP_ID and FEISHU_APP_SECRET must be set")
        sys.exit(1)

    print(f"App ID: {app_id}")
    print("Connecting to Feishu WS... (send any message to the bot in Feishu)")
    print("-" * 60)

    done = threading.Event()
    client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

    def handle_message(data: P2ImMessageReceiveV1) -> None:
        event = data.event
        sender = event.sender
        open_id = sender.sender_id.open_id
        msg_type = event.message.message_type
        content = event.message.content

        print(f"\n{'=' * 60}")
        print("✅ Pairing successful!")
        print(f"   open_id   : {open_id}")
        print(f"   msg_type  : {msg_type}")
        print(f"   content   : {content}")
        print(f"{'=' * 60}")
        print("\nAdd this to your .env file:")
        print(f"   FEISHU_OPEN_ID={open_id}")
        print()

        # Reply to confirm
        try:
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("open_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("text")
                    .content(json.dumps({"text": f"✅ Pairing done! open_id={open_id}"}))
                    .build()
                )
                .build()
            )
            resp = client.im.v1.message.create(req)
            if resp.success():
                print("Reply sent to Feishu.")
            else:
                print(f"Reply failed: {resp.code} {resp.msg}")
        except Exception as e:
            print(f"Reply error: {e}")

        done.set()

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(handle_message)
        .build()
    )

    ws_client = lark.ws.Client(
        app_id=app_id,
        app_secret=app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    # Start WS in a daemon thread so we can exit after pairing
    ws_thread = threading.Thread(target=ws_client.start, daemon=True)
    ws_thread.start()

    # Wait for pairing (timeout 10 min — enough time to configure event subscription)
    if done.wait(timeout=600):
        print("Done! You can now run the smoke test with FEISHU_OPEN_ID set.")
    else:
        print("Timeout: no message received in 10 minutes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
