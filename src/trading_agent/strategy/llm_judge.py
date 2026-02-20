"""LLMJudgeEngineImpl — PRD §4.3.2."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from trading_agent.models import Action, NewsItem, SignalStatus, TradeSignal
from trading_agent.strategy.llm_engine import LLMClient, _KNOWN_FACTORS
from trading_agent.strategy.prompts import (
    TRADING_JUDGE_SYSTEM_PROMPT,
    build_stock_prompt,
)
from trading_agent.tz import now as _now

logger = logging.getLogger(__name__)


class LLMJudgeEngineImpl:
    """LLM comprehensive judge: evidence package → BUY/SELL/HOLD.

    Uses bull/bear/judge three-role debate (inspired by TradingAgents).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    async def judge_stock(
        self,
        symbol: str,
        name: str,
        factor_score: float,
        factor_rank: int,
        factor_details: dict[str, float],
        stock_news: list[NewsItem],
        announcements: list[NewsItem],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        price: float,
        *,
        factor_signal: str = "",
        universe_size: int = 0,
    ) -> TradeSignal:
        """Judge a single stock. Always returns a TradeSignal (never None)."""
        try:
            filtered_details = {
                k: v for k, v in factor_details.items() if k in _KNOWN_FACTORS
            }

            user_prompt = build_stock_prompt(
                symbol=symbol,
                name=name,
                factor_score=factor_score,
                factor_rank=factor_rank,
                factor_details=filtered_details,
                stock_news=stock_news,
                announcements=announcements,
                telegraph=telegraph,
                global_market=global_market,
                factor_signal=factor_signal,
                universe_size=universe_size,
            )

            result = await asyncio.wait_for(
                self._client.call(TRADING_JUDGE_SYSTEM_PROMPT, user_prompt),
                timeout=60,
            )
            return self._parse_signal(
                result, symbol, name, factor_score, factor_rank,
                filtered_details, price, _now(),
            )
        except Exception:  # noqa: BLE001
            logger.warning("LLM judge failed for %s, using factor fallback",
                           symbol, exc_info=True)
            return self._factor_fallback(
                symbol, name, factor_score, price, _now(),
            )

    async def judge_pool(
        self,
        candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        *,
        universe_size: int = 0,
    ) -> list[TradeSignal]:
        """Judge a pool of candidates sequentially.

        Each candidate dict must contain:
            symbol, name, factor_score, factor_rank, factor_details,
            stock_news, announcements, price

        Parameters
        ----------
        universe_size : total number of stocks in the factor universe (for rank display)
        """
        signals: list[TradeSignal] = []
        ok, fail = 0, 0
        pool_size = universe_size or len(candidates)

        for c in candidates:
            try:
                signal = await self.judge_stock(
                    symbol=c["symbol"],
                    name=c["name"],
                    factor_score=c["factor_score"],
                    factor_rank=c["factor_rank"],
                    factor_details=c["factor_details"],
                    stock_news=c.get("stock_news", []),
                    announcements=c.get("announcements", []),
                    telegraph=telegraph,
                    global_market=global_market,
                    price=c["price"],
                    factor_signal=c.get("factor_signal", ""),
                    universe_size=universe_size,
                )
                if signal.metadata is not None:
                    signal.metadata["pool_size"] = pool_size
                signals.append(signal)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("judge_pool: %s failed", c.get("symbol", "UNKNOWN"),
                               exc_info=True)

        logger.info("judge_pool: %d/%d succeeded, %d failed",
                    ok, len(candidates), fail)
        return signals

    def build_evidence_pool(
        self,
        candidate_symbols: list[str],
        pool_df: pd.DataFrame,
        factor_matrix: pd.DataFrame,
        status_map: dict[str, tuple[SignalStatus, int, float | None]],
        tech_df: dict[str, pd.DataFrame],
        name_map: dict[str, str],
        stock_news_map: dict[str, tuple[list, list]],
        *,
        factor_signal_map: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build evidence packages for judge_pool. Pure data in, pure data out.

        Parameters
        ----------
        candidate_symbols : ordered list of symbols to evaluate
        pool_df : factor pool DataFrame (symbol, score, rank)
        factor_matrix : full factor matrix indexed by symbol
        status_map : symbol → (SignalStatus, days_in_top_n, prev_factor_score)
        tech_df : symbol → computed tech DataFrame (with 'close' column)
        name_map : symbol → display name
        stock_news_map : symbol → (stock_news, announcements)
        factor_signal_map : symbol → factor signal label (BUY/EXIT/WATCHLIST)
        """
        _fsm = factor_signal_map or {}
        evidence_pool: list[dict] = []
        for sym in candidate_symbols:
            row = pool_df[pool_df["symbol"] == sym] if not pool_df.empty else pd.DataFrame()
            f_score = float(row["score"].iloc[0]) if not row.empty else 0.5
            f_rank = int(row["rank"].iloc[0]) if not row.empty else len(pool_df) + 1
            f_details = (
                factor_matrix.loc[sym].to_dict()
                if sym in factor_matrix.index else {}
            )
            sig_status, days_in_top, prev_score = status_map.get(
                sym, (SignalStatus.NEW_ENTRY, 0, None),
            )
            s_news, s_anns = stock_news_map.get(sym, ([], []))

            bars = tech_df.get(sym)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0.0

            evidence_pool.append({
                "symbol": sym,
                "name": name_map.get(sym, sym),
                "factor_score": f_score,
                "factor_rank": f_rank,
                "factor_details": f_details,
                "factor_signal": _fsm.get(sym, ""),
                "signal_status": sig_status.value,
                "days_in_top_n": days_in_top,
                "prev_factor_score": prev_score,
                "stock_news": s_news,
                "announcements": s_anns,
                "price": price,
            })
        return evidence_pool

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_signal(
        self,
        result: dict,
        symbol: str,
        name: str,
        factor_score: float,
        factor_rank: int,
        factor_details: dict[str, float],
        price: float,
        now: Any,
    ) -> TradeSignal:
        """Parse LLM JSON output into a TradeSignal."""
        raw_action = str(result.get("action", "HOLD")).upper().strip()
        try:
            action = Action(raw_action)
        except ValueError:
            logger.warning("Invalid action '%s' for %s, defaulting to HOLD",
                           raw_action, symbol)
            action = Action.HOLD

        confidence = float(result.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        bull = result.get("bull_reason", "")
        bear = result.get("bear_reason", "")
        conclusion = result.get("judge_conclusion", "")
        reason = conclusion

        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=confidence,
            source="llm_judge",
            reason=reason,
            factor_score=factor_score,
            metadata={
                "bull_reason": bull,
                "bear_reason": bear,
                "judge_conclusion": conclusion,
                "short_term_outlook": result.get("short_term_outlook", ""),
                "mid_term_outlook": result.get("mid_term_outlook", ""),
                "factor_rank": factor_rank,
                "factor_details": factor_details,
            },
        )

    def _factor_fallback(
        self,
        symbol: str,
        name: str,
        factor_score: float,
        price: float,
        now: Any,
    ) -> TradeSignal:
        """Fallback when LLM is unavailable: use pure factor score."""
        if factor_score >= 0.7:
            action = Action.BUY
        elif factor_score <= 0.3:
            action = Action.SELL
        else:
            action = Action.HOLD

        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=factor_score,
            source="factor_fallback",
            reason=f"LLM 不可用, 纯因子降级 (score={factor_score:.4f})",
            factor_score=factor_score,
            metadata={"fallback": True},
        )
