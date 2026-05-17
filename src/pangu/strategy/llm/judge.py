"""LLMJudgeEngineImpl — pool-level Bull/Bear/Judge rebalance debate."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pangu.models import NewsItem
from pangu.strategy.llm.client import LLMClient
from pangu.strategy.llm.prompts import (
    build_rebalance_prompt,
    build_rebalance_system_prompt,
)

logger = logging.getLogger(__name__)

# Factor keys forwarded to the LLM (whitelist).
_KNOWN_FACTORS: frozenset[str] = frozenset(
    {
        "rsi_14",
        "macd_hist",
        "bias_20",
        "obv",
        "atr_14",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "roe_ttm",
    }
)


# ---------------------------------------------------------------------------
# Rebalance dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Pick:
    """A single LLM-chosen symbol with reason and supporting evidence."""

    symbol: str
    reason: str
    evidence: str = ""


@dataclass
class DebateNotes:
    """Bull / bear debate text captured for a single candidate pool."""

    bull: str = ""
    bear: str = ""


@dataclass
class RebalanceDecision:
    """Outcome of one pool-level Bull/Bear/Judge call."""

    sells: list[Pick] = field(default_factory=list)
    buys: list[Pick] = field(default_factory=list)
    sell_debate: DebateNotes = field(default_factory=DebateNotes)
    buy_debate: DebateNotes = field(default_factory=DebateNotes)
    source: str = "llm_judge"  # "llm_judge" | "llm_failed"


class LLMJudgeEngineImpl:
    """LLM pool-level rebalance judge (Bull / Bear / Judge debate)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    # ------------------------------------------------------------------
    # Rebalance judge (pool-level Bull/Bear/Judge debate)
    # ------------------------------------------------------------------

    async def judge_rebalance(
        self,
        *,
        today: str,
        sell_candidates: list[dict[str, Any]],
        buy_candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        top_n: int,
        n_drop: int,
        universe_size: int = 0,
        timeout: float = 120.0,
    ) -> RebalanceDecision:
        """LLM rebalance decision over SELL + BUY candidate pools.

        Pool-level Bull/Bear/Judge debate: one LLM call analyses both pools
        and selects up to ``n_drop`` symbols to SELL and ``n_drop`` symbols
        to BUY, each with a reason and (optional) evidence string.

        Symbols not in the corresponding pool are silently dropped. If the
        LLM fails, returns an empty RebalanceDecision so the caller can fall
        back to the ML-rank-based TopkDropout.
        """
        sell_pool_syms = {c["symbol"] for c in sell_candidates}
        buy_pool_syms = {c["symbol"] for c in buy_candidates}

        sell_norm = [self._normalize_candidate(c) for c in sell_candidates]
        buy_norm = [self._normalize_candidate(c) for c in buy_candidates]

        if not sell_pool_syms and not buy_pool_syms:
            return RebalanceDecision(source="llm_judge")

        try:
            system_prompt = build_rebalance_system_prompt(
                top_n=top_n,
                n_drop=n_drop,
                sell_pool_size=len(sell_norm),
                buy_pool_size=len(buy_norm),
            )
            user_prompt = build_rebalance_prompt(
                today=today,
                sell_candidates=sell_norm,
                buy_candidates=buy_norm,
                telegraph=telegraph,
                global_market=global_market,
                top_n=top_n,
                n_drop=n_drop,
                universe_size=universe_size,
            )
            result = await asyncio.wait_for(
                self._client.call(system_prompt, user_prompt),
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "judge_rebalance: LLM call failed, returning empty decision",
                exc_info=True,
            )
            return RebalanceDecision(source="llm_failed")

        return self._parse_rebalance(
            result,
            sell_pool=sell_pool_syms,
            buy_pool=buy_pool_syms,
            n_drop=n_drop,
        )

    # ------------------------------------------------------------------
    # Rebalance helpers
    # ------------------------------------------------------------------

    def _normalize_candidate(self, cand: dict[str, Any]) -> dict[str, Any]:
        """Filter factor_details to whitelist and ensure required keys exist."""
        details = cand.get("factor_details") or {}
        filtered = {k: v for k, v in details.items() if k in _KNOWN_FACTORS}
        return {
            "symbol": cand.get("symbol", ""),
            "name": cand.get("name", cand.get("symbol", "")),
            "ml_score": float(cand.get("ml_score", 0.0)),
            "ml_rank": int(cand.get("ml_rank", 0)),
            "prev_ml_rank": cand.get("prev_ml_rank"),
            "rank_delta": cand.get("rank_delta"),
            "factor_details": filtered,
            "stock_news": cand.get("stock_news") or [],
            "announcements": cand.get("announcements") or [],
        }

    def _parse_rebalance(
        self,
        result: dict,
        *,
        sell_pool: set[str],
        buy_pool: set[str],
        n_drop: int,
    ) -> RebalanceDecision:
        """Parse LLM JSON into a RebalanceDecision, enforcing pool / count rules."""
        sells_raw = result.get("sells") or []
        buys_raw = result.get("buys") or []

        sell_debate = self._parse_debate_notes(result.get("sell_debate"))
        buy_debate = self._parse_debate_notes(result.get("buy_debate"))

        sells = self._parse_picks(sells_raw, allowed=sell_pool, n_drop=n_drop)
        buys = self._parse_picks(buys_raw, allowed=buy_pool, n_drop=n_drop)

        return RebalanceDecision(
            sells=sells,
            buys=buys,
            sell_debate=sell_debate,
            buy_debate=buy_debate,
            source="llm_judge",
        )

    @staticmethod
    def _parse_debate_notes(raw: Any) -> DebateNotes:
        if not isinstance(raw, dict):
            return DebateNotes()
        return DebateNotes(
            bull=str(raw.get("bull", "") or ""),
            bear=str(raw.get("bear", "") or ""),
        )

    @staticmethod
    def _parse_picks(
        raw: Any,
        *,
        allowed: set[str],
        n_drop: int,
    ) -> list[Pick]:
        """Deduplicate, filter by pool membership, cap at n_drop."""
        if not isinstance(raw, list):
            return []
        seen: set[str] = set()
        picks: list[Pick] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", "")).strip()
            if not sym or sym in seen or sym not in allowed:
                continue
            reason = str(item.get("reason", "") or "").strip()
            if not reason:
                continue
            evidence = str(item.get("evidence", "") or "").strip()
            picks.append(Pick(symbol=sym, reason=reason, evidence=evidence))
            seen.add(sym)
            if len(picks) >= n_drop:
                break
        return picks
