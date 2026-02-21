"""SQLite storage layer — PRD §7.1.

Provides a thin wrapper around sqlite3 for persisting daily bars,
news items, trade signals, fundamentals, data sync log, and the
trading calendar.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from pangu.models import (
    Action,
    NewsCategory,
    NewsItem,
    Region,
    SignalStatus,
    TradeSignal,
)
from pangu.tz import now as _now

# ---------------------------------------------------------------------------
# DDL — per PRD §7.1
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,       -- YYYY-MM-DD
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    amount      REAL,
    adj_factor  REAL DEFAULT 1.0,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS news_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT,
    source      TEXT,
    region      TEXT DEFAULT 'domestic',
    category    TEXT DEFAULT 'news',
    symbols     TEXT,               -- JSON array
    sentiment   REAL,
    raw_json    TEXT,
    title_hash  TEXT,
    UNIQUE (source, title_hash)
);

CREATE TABLE IF NOT EXISTS trade_signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    name             TEXT,
    action           TEXT NOT NULL,
    signal_status    TEXT,
    days_in_top_n    INTEGER DEFAULT 0,
    price            REAL,
    confidence       REAL,
    source           TEXT,
    reason           TEXT,
    stop_loss        REAL,
    take_profit      REAL,
    factor_score     REAL,
    prev_factor_score REAL,
    pushed           INTEGER DEFAULT 0,
    metadata         TEXT,
    signal_date      TEXT,
    return_1d        REAL,
    return_3d        REAL,
    return_5d        REAL,
    UNIQUE (symbol, action, signal_date)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_time        TEXT NOT NULL,
    strategy_name   TEXT,
    config_json     TEXT,
    sharpe_ratio    REAL,
    annual_return   REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    total_trades    INTEGER,
    report_path     TEXT
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol       TEXT NOT NULL,
    date         TEXT NOT NULL,
    pe_ttm       REAL,
    pb           REAL,
    roe_ttm      REAL,
    revenue_yoy  REAL,
    profit_yoy   REAL,
    market_cap   REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS data_sync_log (
    symbol      TEXT NOT NULL,
    data_type   TEXT NOT NULL,
    last_sync   TEXT NOT NULL,
    last_date   TEXT,
    source      TEXT NOT NULL,
    status      TEXT NOT NULL,
    error_msg   TEXT,
    PRIMARY KEY (symbol, data_type)
);

CREATE TABLE IF NOT EXISTS trading_calendar (
    date TEXT PRIMARY KEY   -- YYYY-MM-DD
);

CREATE TABLE IF NOT EXISTS global_snapshots (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,       -- YYYY-MM-DD
    name        TEXT,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    change_pct  REAL,
    source      TEXT,               -- 'us_index', 'hk_index', 'commodity'
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS factor_pool (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,       -- YYYY-MM-DD
    score       REAL,
    rank        INTEGER,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS index_constituents (
    symbol        TEXT NOT NULL,
    name          TEXT,
    index_code    TEXT NOT NULL,     -- e.g. '000300' for CSI300
    sector        TEXT,              -- 东财行业分类
    updated_date  TEXT NOT NULL,     -- YYYY-MM-DD
    PRIMARY KEY (symbol, index_code)
);
"""


class Database:
    """Thin SQLite wrapper for PanGu persistence."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """Create all tables if they don't exist."""
        with self._lock:
            self._conn.executescript(_DDL)
            self._migrate()

    def _migrate(self) -> None:
        """Apply schema migrations for existing databases."""
        # Add category column to news_items if missing
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(news_items)").fetchall()
        }
        if "category" not in cols:
            self._conn.execute(
                "ALTER TABLE news_items ADD COLUMN category TEXT DEFAULT 'news'"
            )
            self._conn.commit()

        # Migrate trade_signals: add signal_date + UNIQUE constraint
        sig_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(trade_signals)").fetchall()
        }
        if "signal_date" not in sig_cols:
            # SQLite can't ALTER TABLE ADD UNIQUE, so recreate the table
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS trade_signals_new (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT NOT NULL,
                    symbol           TEXT NOT NULL,
                    name             TEXT,
                    action           TEXT NOT NULL,
                    signal_status    TEXT,
                    days_in_top_n    INTEGER DEFAULT 0,
                    price            REAL,
                    confidence       REAL,
                    source           TEXT,
                    reason           TEXT,
                    stop_loss        REAL,
                    take_profit      REAL,
                    factor_score     REAL,
                    prev_factor_score REAL,
                    pushed           INTEGER DEFAULT 0,
                    metadata         TEXT,
                    signal_date      TEXT,
                    UNIQUE (symbol, action, signal_date)
                );
                INSERT OR IGNORE INTO trade_signals_new
                    (timestamp, symbol, name, action, signal_status, days_in_top_n,
                     price, confidence, source, reason, stop_loss, take_profit,
                     factor_score, prev_factor_score, pushed, metadata, signal_date)
                SELECT timestamp, symbol, name, action, signal_status, days_in_top_n,
                       price, confidence, source, reason, stop_loss, take_profit,
                       factor_score, prev_factor_score, pushed, metadata,
                       substr(timestamp, 1, 10)
                FROM trade_signals;
                DROP TABLE trade_signals;
                ALTER TABLE trade_signals_new RENAME TO trade_signals;
            """)
            # Re-read columns after migration
            sig_cols = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(trade_signals)").fetchall()
            }

        # Migrate trade_signals: add return_1d/3d/5d columns
        for col in ("return_1d", "return_3d", "return_5d"):
            if col not in sig_cols:
                self._conn.execute(f"ALTER TABLE trade_signals ADD COLUMN {col} REAL")
        self._conn.commit()

    # ------------------------------------------------------------------
    # daily_bars
    # ------------------------------------------------------------------

    def save_daily_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE daily bars from *df*.

        *df* must contain columns: date, open, high, low, close, volume.
        Optional: amount, adj_factor.  Returns row count inserted.
        """
        if df.empty:
            return 0

        rows: list[tuple[Any, ...]] = []
        for _, r in df.iterrows():
            rows.append((
                symbol,
                str(r["date"]),
                r.get("open"),
                r.get("high"),
                r.get("low"),
                r.get("close"),
                r.get("volume"),
                r.get("amount"),
                r.get("adj_factor", 1.0),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO daily_bars "
                "(symbol, date, open, high, low, close, volume, amount, adj_factor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def load_daily_bars(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load daily bars for *symbol* between *start* and *end* (inclusive)."""
        with self._lock:
            return pd.read_sql(
                "SELECT * FROM daily_bars WHERE symbol = ? AND date >= ? AND date <= ? "
                "ORDER BY date",
                self._conn,
                params=(symbol, start, end),
            )

    # ------------------------------------------------------------------
    # data_sync_log
    # ------------------------------------------------------------------

    def get_last_sync_date(self, symbol: str, data_type: str) -> str | None:
        """Return the last data date for *(symbol, data_type)*, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT last_date FROM data_sync_log WHERE symbol = ? AND data_type = ?",
                (symbol, data_type),
            ).fetchone()
        return row[0] if row else None

    def update_sync_log(
        self,
        symbol: str,
        data_type: str,
        status: str,
        source: str,
        *,
        last_date: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        """Upsert a row in data_sync_log."""
        now = _now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO data_sync_log "
                "(symbol, data_type, last_sync, last_date, source, status, error_msg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, data_type, now, last_date, source, status, error_msg),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # news_items
    # ------------------------------------------------------------------

    @staticmethod
    def _title_hash(title: str) -> str:
        return hashlib.sha256(title.encode()).hexdigest()[:16]

    def save_news_items(self, items: list[NewsItem]) -> int:
        """Insert news items, dedup by (source, title_hash). Returns count inserted."""
        inserted = 0
        with self._lock:
            for item in items:
                try:
                    self._conn.execute(
                        "INSERT INTO news_items "
                        "(timestamp, title, content, source, region, category, symbols, sentiment, title_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            item.timestamp.isoformat(),
                            item.title,
                            item.content,
                            item.source,
                            item.region.value,
                            item.category.value,
                            json.dumps(item.symbols),
                            item.sentiment,
                            self._title_hash(item.title),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            self._conn.commit()
        return inserted

    def load_recent_news(self, hours: int = 24) -> list[NewsItem]:
        """Load news items from the last *hours* hours."""
        cutoff = (
            _now() - timedelta(hours=hours)
        ).isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, title, content, source, region, category, symbols, sentiment "
                "FROM news_items WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
        result: list[NewsItem] = []
        for ts, title, content, source, region, category, symbols_json, sentiment in rows:
            result.append(
                NewsItem(
                    timestamp=datetime.fromisoformat(ts),
                    title=title,
                    content=content or "",
                    source=source or "",
                    region=Region(region),
                    symbols=json.loads(symbols_json) if symbols_json else [],
                    sentiment=sentiment,
                    category=NewsCategory(category) if category else NewsCategory.NEWS,
                )
            )
        return result

    def cleanup_old_news(self, days: int = 30) -> int:
        """Delete news items older than *days* days. Returns count deleted."""
        cutoff = (_now() - timedelta(days=days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM news_items WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # trade_signals
    # ------------------------------------------------------------------

    def save_trade_signal(self, signal: TradeSignal) -> int:
        """Insert or replace a trade signal. Dedup by (symbol, action, date).

        Returns the inserted/replaced row id.
        """
        signal_date = signal.timestamp.strftime("%Y-%m-%d")
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR REPLACE INTO trade_signals "
                "(timestamp, symbol, name, action, signal_status, days_in_top_n, "
                "price, confidence, source, reason, "
                "stop_loss, take_profit, factor_score, prev_factor_score, pushed, metadata, signal_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.name,
                    signal.action.value,
                    signal.signal_status.value,
                    signal.days_in_top_n,
                    signal.price,
                    signal.confidence,
                    signal.source,
                    signal.reason,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.factor_score,
                    signal.prev_factor_score,
                    json.dumps(signal.metadata),
                    signal_date,
                ),
            )
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def load_signals(self, date: str) -> list[TradeSignal]:
        """Load all signals whose timestamp starts with *date* (YYYY-MM-DD)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, symbol, name, action, signal_status, days_in_top_n, "
                "price, confidence, source, reason, "
                "stop_loss, take_profit, factor_score, prev_factor_score, metadata "
                "FROM trade_signals WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{date}%",),
            ).fetchall()
        result: list[TradeSignal] = []
        for (
            ts, symbol, name, action, signal_status, days_in_top_n,
            price, confidence, source, reason,
            stop_loss, take_profit, factor_score, prev_factor_score, metadata_json,
        ) in rows:
            result.append(
                TradeSignal(
                    timestamp=datetime.fromisoformat(ts),
                    symbol=symbol,
                    name=name or "",
                    action=Action(action),
                    signal_status=SignalStatus(signal_status) if signal_status else SignalStatus.NEW_ENTRY,
                    days_in_top_n=days_in_top_n or 0,
                    price=price or 0.0,
                    confidence=confidence or 0.0,
                    source=source or "",
                    reason=reason or "",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    factor_score=factor_score,
                    prev_factor_score=prev_factor_score,
                    metadata=json.loads(metadata_json) if metadata_json else {},
                )
            )
        return result

    def load_unverified_signals(self, signal_date: str, lookback: int) -> list[dict]:
        """Load signals from *signal_date* where return_{lookback}d is NULL.

        Returns list of dicts with keys: id, symbol, name, action, price.
        Only BUY/SELL signals are returned (HOLD is not verifiable).
        """
        col = f"return_{lookback}d"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, symbol, name, action, price FROM trade_signals "
                f"WHERE signal_date = ? AND {col} IS NULL "
                f"AND action IN ('BUY', 'SELL')",
                (signal_date,),
            ).fetchall()
        return [
            {"id": r[0], "symbol": r[1], "name": r[2] or r[1], "action": r[3], "price": r[4]}
            for r in rows
        ]

    def update_signal_return(self, signal_id: int, lookback: int, return_pct: float) -> None:
        """Update return_{lookback}d for a signal."""
        col = f"return_{lookback}d"
        with self._lock:
            self._conn.execute(
                f"UPDATE trade_signals SET {col} = ? WHERE id = ?",
                (return_pct, signal_id),
            )
            self._conn.commit()

    def get_signal_returns(self, days: int = 30) -> dict[str, dict]:
        """Compute strategy return stats over the last *days* days.

        Returns ``{lookback: {"count": N, "avg_return": pct}}``.
        Strategy return = return_pct for BUY, -return_pct for SELL.
        """
        results: dict[str, dict] = {}
        for lb in (1, 3, 5):
            col = f"return_{lb}d"
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT action, {col} FROM trade_signals "
                    f"WHERE {col} IS NOT NULL "
                    f"AND signal_date >= date('now', ?)",
                    (f"-{days} days",),
                ).fetchall()
            if not rows:
                results[f"{lb}d"] = {"count": 0, "avg_return": 0.0}
                continue
            total_return = sum(
                ret if action == "BUY" else -ret
                for action, ret in rows
            )
            results[f"{lb}d"] = {
                "count": len(rows),
                "avg_return": round(total_return / len(rows), 2),
            }
        return results

    # ------------------------------------------------------------------
    # trading_calendar
    # ------------------------------------------------------------------

    def save_trading_calendar(self, dates: list[str]) -> int:
        """Insert trading calendar dates (YYYY-MM-DD). Returns count of *new* rows inserted."""
        with self._lock:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO trading_calendar (date) VALUES (?)",
                [(d,) for d in dates],
            )
            self._conn.commit()
        return cur.rowcount

    def is_trading_day(self, date: str) -> bool:
        """Check if *date* is a trading day."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM trading_calendar WHERE date = ?", (date,)
            ).fetchone()
        return row is not None

    def get_trading_day_offset(self, date: str, offset: int) -> str | None:
        """Return the trading day *offset* days before *date* (offset > 0 means past).

        E.g. ``get_trading_day_offset("2026-02-21", 3)`` returns the 3rd trading
        day before 2026-02-21.  Returns None if not enough calendar data.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT date FROM trading_calendar WHERE date < ? ORDER BY date DESC LIMIT ?",
                (date, offset),
            ).fetchall()
        if len(rows) < offset:
            return None
        return rows[-1][0]

    def has_trading_day_between(self, after: str, up_to: str) -> bool | None:
        """Return True if a trading day exists in (after, up_to].

        Returns None when the calendar table is empty (caller should fallback).
        """
        with self._lock:
            has_cal = self._conn.execute("SELECT 1 FROM trading_calendar LIMIT 1").fetchone()
            if has_cal is None:
                return None
            row = self._conn.execute(
                "SELECT 1 FROM trading_calendar WHERE date > ? AND date <= ? LIMIT 1",
                (after, up_to),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # db stats
    # ------------------------------------------------------------------

    def get_db_stats(self) -> dict:
        """Return summary counts for the status CLI command."""
        with self._lock:
            bars_count = self._conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
            bars_symbols = self._conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_bars").fetchone()[0]
            signals_count = self._conn.execute("SELECT COUNT(*) FROM trade_signals").fetchone()[0]
            calendar_count = self._conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
            news_count = self._conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
            latest_signal = self._conn.execute(
                "SELECT signal_date FROM trade_signals ORDER BY signal_date DESC LIMIT 1"
            ).fetchone()
            latest_bar = self._conn.execute(
                "SELECT MAX(date) FROM daily_bars"
            ).fetchone()
        return {
            "bars_count": bars_count,
            "bars_symbols": bars_symbols,
            "signals_count": signals_count,
            "calendar_count": calendar_count,
            "news_count": news_count,
            "latest_signal": latest_signal[0] if latest_signal else "N/A",
            "latest_bar": latest_bar[0] if latest_bar else "N/A",
        }

    # ------------------------------------------------------------------
    # fundamentals
    # ------------------------------------------------------------------

    def save_fundamentals(self, symbol: str, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE fundamentals rows from *df*. Returns row count."""
        if df.empty:
            return 0
        rows: list[tuple[Any, ...]] = []
        for _, r in df.iterrows():
            rows.append((
                symbol,
                str(r["date"]),
                r.get("pe_ttm"),
                r.get("pb"),
                r.get("roe_ttm"),
                r.get("revenue_yoy"),
                r.get("profit_yoy"),
                r.get("market_cap"),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO fundamentals "
                "(symbol, date, pe_ttm, pb, roe_ttm, revenue_yoy, profit_yoy, market_cap) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def load_fundamentals(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load fundamentals for *symbol* between *start* and *end*."""
        with self._lock:
            return pd.read_sql(
                "SELECT * FROM fundamentals WHERE symbol = ? AND date >= ? AND date <= ? "
                "ORDER BY date",
                self._conn,
                params=(symbol, start, end),
            )

    # ------------------------------------------------------------------
    # global_snapshots
    # ------------------------------------------------------------------

    def save_global_snapshots(self, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE global market snapshots. Returns row count.

        *df* must contain columns: symbol, date.
        Optional: name, open, high, low, close, volume, change_pct, source.
        """
        if df.empty:
            return 0
        rows: list[tuple[Any, ...]] = []
        for _, r in df.iterrows():
            rows.append((
                r["symbol"],
                str(r["date"]),
                r.get("name"),
                r.get("open"),
                r.get("high"),
                r.get("low"),
                r.get("close"),
                r.get("volume"),
                r.get("change_pct"),
                r.get("source"),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO global_snapshots "
                "(symbol, date, name, open, high, low, close, volume, change_pct, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def load_latest_global_snapshots(self) -> pd.DataFrame:
        """Load the most recent snapshot for each symbol."""
        with self._lock:
            return pd.read_sql(
                "SELECT g.* FROM global_snapshots g "
                "INNER JOIN ("
                "  SELECT symbol, MAX(date) AS max_date FROM global_snapshots GROUP BY symbol"
                ") m ON g.symbol = m.symbol AND g.date = m.max_date "
                "ORDER BY g.source, g.symbol",
                self._conn,
            )

    # ------------------------------------------------------------------
    # Factor pool
    # ------------------------------------------------------------------

    def save_factor_pool(self, date: str, df: pd.DataFrame) -> int:
        """Save factor scores and ranks for a given date.

        *df* must have columns: symbol, score, rank.
        """
        if df.empty:
            return 0
        rows = [
            (row["symbol"], date, float(row["score"]), int(row["rank"]))
            for _, row in df.iterrows()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO factor_pool (symbol, date, score, rank) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def load_factor_pool(self, date: str) -> pd.DataFrame:
        """Load factor pool for a specific date."""
        with self._lock:
            return pd.read_sql(
                "SELECT symbol, score, rank FROM factor_pool WHERE date = ? ORDER BY rank",
                self._conn,
                params=(date,),
            )

    def load_factor_pool_latest(self) -> pd.DataFrame:
        """Load factor pool for the most recent date."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(date) FROM factor_pool"
            ).fetchone()
            if row is None or row[0] is None:
                return pd.DataFrame(columns=["symbol", "score", "rank"])
            return pd.read_sql(
                "SELECT symbol, score, rank FROM factor_pool WHERE date = ? ORDER BY rank",
                self._conn,
                params=(row[0],),
            )

    def load_factor_pool_previous_day(self) -> pd.DataFrame:
        """Load factor pool for the most recent date *before* today."""
        from pangu.utils import date_str
        today = date_str()
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(date) FROM factor_pool WHERE date < ?", (today,)
            ).fetchone()
            if row is None or row[0] is None:
                return pd.DataFrame(columns=["symbol", "score", "rank"])
            return pd.read_sql(
                "SELECT symbol, score, rank FROM factor_pool WHERE date = ? ORDER BY rank",
                self._conn,
                params=(row[0],),
            )

    # ------------------------------------------------------------------
    # Index constituents
    # ------------------------------------------------------------------

    def save_index_constituents(self, rows: list[dict]) -> int:
        """INSERT OR REPLACE index constituents.

        Each dict must have: symbol, name, index_code, updated_date.
        Optional: sector.
        """
        if not rows:
            return 0
        tuples = [
            (r["symbol"], r.get("name"), r["index_code"],
             r.get("sector"), r["updated_date"])
            for r in rows
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO index_constituents "
                "(symbol, name, index_code, sector, updated_date) "
                "VALUES (?, ?, ?, ?, ?)",
                tuples,
            )
            self._conn.commit()
        return len(tuples)

    def load_index_constituents(self, index_code: str) -> list[dict]:
        """Load constituents for a given index code."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, name, index_code, sector, updated_date "
                "FROM index_constituents WHERE index_code = ? ORDER BY symbol",
                (index_code,),
            ).fetchall()
        return [
            {"symbol": r[0], "name": r[1], "index_code": r[2],
             "sector": r[3], "updated_date": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
