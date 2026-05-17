"""Target portfolio state — persists the system's virtual holdings.

The production pipeline does not own real money, but it must remember which
stocks the recommended portfolio currently holds in order to:

* compute SELL candidate pool (worst-ranked of current holdings)
* compute BUY candidate pool (best-ranked of non-held)
* expose rank-delta context to the LLM

Persistence format: JSON written atomically via ``tmp + os.replace`` to avoid
truncation if the process crashes mid-write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Virtual portfolio snapshot at a given decision date."""

    date: str
    symbols: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


class PortfolioState:
    """Read / write the latest target portfolio JSON file.

    The file holds a single Portfolio (the latest snapshot). Each rebalance
    overwrites the file; we do not keep history here (DB ``trade_signals``
    already captures every rebalance decision).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Portfolio | None:
        """Load latest portfolio. Returns None for cold start (missing file).

        Returns None and logs a warning when the file exists but is corrupt
        (so the caller falls back to cold start rather than crashing).
        """
        if not self._path.exists():
            return None
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Portfolio state file %s is unreadable or corrupt; treating as cold start",
                self._path,
                exc_info=True,
            )
            return None
        try:
            return Portfolio(
                date=str(data["date"]),
                symbols=list(data.get("symbols", [])),
                scores={str(k): float(v) for k, v in (data.get("scores") or {}).items()},
                ranks={str(k): int(v) for k, v in (data.get("ranks") or {}).items()},
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Portfolio state file %s has unexpected schema; treating as cold start",
                self._path,
                exc_info=True,
            )
            return None

    def save(self, portfolio: Portfolio) -> None:
        """Atomically write portfolio to disk (tmp file + rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(portfolio)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_path = tmp.name
        os.replace(tmp_path, self._path)
        logger.info(
            "PortfolioState saved %d symbols to %s (date=%s)",
            len(portfolio.symbols),
            self._path,
            portfolio.date,
        )

    def clear(self) -> None:
        """Remove the portfolio file (forces next run into cold start)."""
        if self._path.exists():
            self._path.unlink()
            logger.info("PortfolioState cleared: %s", self._path)
