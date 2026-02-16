"""SignalMerger Protocol + SimpleSignalMerger — PRD §4.3.4 / §6."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Protocol

from trading_agent.models import Action, TradeSignal


class SignalMerger(Protocol):
    """Interface for merging signals from multiple strategy engines."""

    def merge(
        self,
        factor_signals: list[TradeSignal],
        event_signals: list[TradeSignal],
        anomaly_signals: list[TradeSignal],
        news_availability: dict[str, bool],
    ) -> list[TradeSignal]:
        """Merge and arbitrate signals using weighted scoring."""
        ...


# ---------------------------------------------------------------------------
# Simple rule-based implementation
# ---------------------------------------------------------------------------

# Confidence boost/penalty for conflict arbitration
_CONFIDENCE_BOOST = 0.15
_CONFLICT_PENALTY = 0.20


def _signal_score(signal: TradeSignal) -> float:
    """Map a signal's action + confidence to a [0, 1] score.

    BUY  → 0.5 + confidence/2  (range 0.5‒1.0)
    SELL → 0.5 - confidence/2  (range 0.0‒0.5)
    HOLD → 0.5
    """
    if signal.action is Action.BUY:
        return 0.5 + signal.confidence / 2
    if signal.action is Action.SELL:
        return 0.5 - signal.confidence / 2
    return 0.5


def _best_signal(signals: list[TradeSignal]) -> TradeSignal | None:
    """Pick the highest-confidence signal from a list (or None)."""
    if not signals:
        return None
    return max(signals, key=lambda s: s.confidence)


def _dedup_key(s: TradeSignal) -> tuple[str, str, str, str]:
    """Dedup key: (symbol, action, source, date)."""
    return (s.symbol, s.action.value, s.source, s.timestamp.strftime("%Y-%m-%d"))


class SimpleSignalMerger:
    """Rule-based signal merger per PRD §4.3.4.

    Parameters come from ``[strategy]`` section of settings.toml.
    """

    def __init__(
        self,
        *,
        factor_weight: float = 0.6,
        event_weight: float = 0.4,
        no_news_factor_weight: float = 0.9,
        no_news_anomaly_weight: float = 0.1,
        announcement_only_factor: float = 0.7,
        announcement_only_event: float = 0.2,
        announcement_only_anomaly: float = 0.1,
        buy_threshold: float = 0.7,
        sell_threshold: float = 0.3,
    ) -> None:
        self.factor_weight = factor_weight
        self.event_weight = event_weight
        self.no_news_factor_weight = no_news_factor_weight
        self.no_news_anomaly_weight = no_news_anomaly_weight
        self.announcement_only_factor = announcement_only_factor
        self.announcement_only_event = announcement_only_event
        self.announcement_only_anomaly = announcement_only_anomaly
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    # -----------------------------------------------------------------
    # Public API (satisfies SignalMerger Protocol)
    # -----------------------------------------------------------------

    def merge(
        self,
        factor_signals: list[TradeSignal],
        event_signals: list[TradeSignal],
        anomaly_signals: list[TradeSignal],
        news_availability: dict[str, bool],
    ) -> list[TradeSignal]:
        # Group signals by (symbol, date)
        factor_by_key = _group_by_symbol_date(factor_signals)
        event_by_key = _group_by_symbol_date(event_signals)
        anomaly_by_key = _group_by_symbol_date(anomaly_signals)

        all_keys = set(factor_by_key) | set(event_by_key) | set(anomaly_by_key)

        results: list[TradeSignal] = []
        seen: set[tuple[str, str, str, str]] = set()

        for key in sorted(all_keys):
            symbol = key[0]
            f_sig = _best_signal(factor_by_key.get(key, []))
            e_sig = _best_signal(event_by_key.get(key, []))
            a_sig = _best_signal(anomaly_by_key.get(key, []))

            merged = self._merge_one(
                symbol, f_sig, e_sig, a_sig,
                has_news=news_availability.get(symbol, False),
            )
            if merged is None:
                continue

            key = _dedup_key(merged)
            if key in seen:
                continue
            seen.add(key)
            results.append(merged)

        return results

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _merge_one(
        self,
        symbol: str,
        f_sig: TradeSignal | None,
        e_sig: TradeSignal | None,
        a_sig: TradeSignal | None,
        *,
        has_news: bool,
    ) -> TradeSignal | None:
        """Merge signals for a single symbol."""
        if f_sig is None and e_sig is None and a_sig is None:
            return None

        # Determine weight mode
        has_event = e_sig is not None
        if has_news and has_event:
            fw, ew, aw = self.factor_weight, self.event_weight, 0.0
        elif has_event:  # announcement only
            fw, ew, aw = (
                self.announcement_only_factor,
                self.announcement_only_event,
                self.announcement_only_anomaly,
            )
        else:  # no news, no event
            fw, ew, aw = self.no_news_factor_weight, 0.0, self.no_news_anomaly_weight

        # Compute weighted score
        total_weight = 0.0
        weighted_sum = 0.0

        if f_sig is not None:
            weighted_sum += fw * _signal_score(f_sig)
            total_weight += fw
        if e_sig is not None:
            weighted_sum += ew * _signal_score(e_sig)
            total_weight += ew
        if a_sig is not None:
            weighted_sum += aw * _signal_score(a_sig)
            total_weight += aw

        if total_weight == 0:
            return None

        merged_score = weighted_sum / total_weight

        # Conflict arbitration — adjust confidence
        # NOTE: per PRD §4.3.4, conflict is only between factor and event.
        # Anomaly is a weak auxiliary signal (max 0.1 weight) and does not
        # participate in arbitration.
        confidence_adj = 0.0
        if f_sig is not None and e_sig is not None:
            if f_sig.action == e_sig.action and f_sig.action is not Action.HOLD:
                confidence_adj = _CONFIDENCE_BOOST  # resonance
            elif (
                f_sig.action is not Action.HOLD
                and e_sig.action is not Action.HOLD
                and f_sig.action != e_sig.action
            ):
                # Conflict → force HOLD
                merged_score = 0.5
                confidence_adj = -_CONFLICT_PENALTY

        # Determine action from merged score
        if merged_score > self.buy_threshold:
            action = Action.BUY
        elif merged_score < self.sell_threshold:
            action = Action.SELL
        else:
            action = Action.HOLD

        # Pick the best available signal as template
        template = f_sig or e_sig or a_sig
        assert template is not None

        base_confidence = max(
            (s.confidence for s in (f_sig, e_sig, a_sig) if s is not None),
            default=0.5,
        )
        confidence = max(0.0, min(1.0, base_confidence + confidence_adj))

        # Build reason
        parts: list[str] = []
        if f_sig is not None:
            parts.append(f"factor({f_sig.action.value})")
        if e_sig is not None:
            parts.append(f"event({e_sig.action.value})")
        if a_sig is not None:
            parts.append(f"anomaly({a_sig.action.value})")
        reason = f"merged(score={merged_score:.2f}): {' + '.join(parts)}"

        return replace(
            template,
            action=action,
            confidence=confidence,
            source="merged",
            reason=reason,
            factor_score=merged_score,
            metadata={
                "merged_score": merged_score,
                "weight_mode": (
                    "news" if has_news and has_event
                    else "announcement" if has_event
                    else "no_news"
                ),
                **(template.metadata or {}),
            },
        )


def _group_by_symbol_date(
    signals: list[TradeSignal],
) -> dict[tuple[str, str], list[TradeSignal]]:
    """Group signals by (symbol, date_str) to preserve multi-day data."""
    groups: dict[tuple[str, str], list[TradeSignal]] = defaultdict(list)
    for s in signals:
        key = (s.symbol, s.timestamp.strftime("%Y-%m-%d"))
        groups[key].append(s)
    return groups
