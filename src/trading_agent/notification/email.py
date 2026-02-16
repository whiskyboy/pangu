"""Email notifier — SMTP_SSL based notification channel."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from trading_agent.models import Action, SignalStatus, TradeSignal

logger = logging.getLogger(__name__)

_ACTION_LABEL = {
    Action.BUY: ("买入信号", "#22c55e"),
    Action.SELL: ("卖出信号", "#ef4444"),
    Action.HOLD: ("持有观望", "#9ca3af"),
}

_STATUS_LABEL = {
    SignalStatus.NEW_ENTRY: "⚡ 首次入选",
    SignalStatus.SUSTAINED: "🔄 维持关注",
    SignalStatus.EXIT: "⛔ 退出",
}


def format_signal_html(signal: TradeSignal) -> str:
    """Build an HTML email body for *signal*."""
    label, color = _ACTION_LABEL.get(signal.action, ("信号", "#666"))
    ts = signal.timestamp.strftime("%Y-%m-%d %H:%M")
    status = _STATUS_LABEL.get(signal.signal_status, signal.signal_status.value)
    action_verb = "买入" if signal.action is Action.BUY else "卖出" if signal.action is Action.SELL else "观望"

    rows = [
        f"<tr><td><b>股票</b></td><td>{signal.name} ({signal.symbol})</td></tr>",
        f"<tr><td><b>信号状态</b></td><td>{status}</td></tr>",
        f"<tr><td><b>建议操作</b></td><td>以 ¥{signal.price:,.2f} {action_verb}</td></tr>",
        f"<tr><td><b>置信度</b></td><td>{signal.confidence:.0%}</td></tr>",
    ]
    if signal.factor_score is not None:
        rows.append(f"<tr><td><b>因子评分</b></td><td>{signal.factor_score:.2f}</td></tr>")
    if signal.stop_loss is not None:
        rows.append(f"<tr><td><b>止损价</b></td><td>¥{signal.stop_loss:,.2f}</td></tr>")
    if signal.take_profit is not None:
        rows.append(f"<tr><td><b>止盈价</b></td><td>¥{signal.take_profit:,.2f}</td></tr>")
    if signal.reason:
        rows.append(f"<tr><td><b>分析</b></td><td>{signal.reason}</td></tr>")

    table = "\n".join(rows)

    return f"""<html><body>
<h2 style="color:{color}">{label} | {ts}</h2>
<table style="border-collapse:collapse" cellpadding="6">
{table}
</table>
<p style="color:#999;font-size:12px">来源: {signal.source} | TradingAgent</p>
</body></html>"""


class EmailNotifier:
    """Send trade signals via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        to_addresses: list[str],
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._to = to_addresses

    async def send(self, signal: TradeSignal) -> bool:
        label, _ = _ACTION_LABEL.get(signal.action, ("信号", ""))
        subject = f"[TradingAgent] {label}: {signal.name} ({signal.symbol})"
        html = format_signal_html(signal)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._user
        msg["To"] = ", ".join(self._to)
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP_SSL(self._host, self._port) as server:
                server.login(self._user, self._password)
                server.sendmail(self._user, self._to, msg.as_string())
            logger.info("Email sent to %s", self._to)
            return True
        except Exception:
            logger.exception("Email send failed")
            return False
