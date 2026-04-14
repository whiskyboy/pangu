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
from datetime import datetime, timedelta
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
    turn        REAL,               -- turnover rate (%)
    preclose    REAL,               -- previous close
    tradestatus TEXT,               -- 1=normal, 0=suspended
    is_st       INTEGER,            -- 1=ST, 0=normal
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
    symbol              TEXT NOT NULL,
    date                TEXT NOT NULL,
    pe_ttm              REAL,
    pb                  REAL,
    roe_ttm             REAL,
    revenue_yoy         REAL,
    profit_yoy          REAL,
    market_cap          REAL,
    net_profit_margin   REAL,
    gross_margin        REAL,
    debt_ratio          REAL,
    asset_turnover      REAL,
    current_ratio       REAL,
    equity_yoy          REAL,
    asset_yoy           REAL,
    cashflow_per_share  REAL,
    cashflow_to_profit  REAL,
    ps_ttm              REAL,
    pcf_ttm             REAL,
    pub_date            TEXT,
    roa                 REAL,
    operating_profit_ratio REAL,
    ocf_to_revenue      REAL,
    eps_weighted         REAL,
    quick_ratio          REAL,
    receivables_turnover REAL,
    inventory_turnover   REAL,
    cost_profit_ratio    REAL,
    dividend_payout_ratio REAL,
    cash_ratio           REAL,
    equity_ratio         REAL,
    shareholder_equity_ratio REAL,
    undistributed_per_share REAL,
    capital_reserve_per_share REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fundamentals_raw (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    data    TEXT NOT NULL,
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
    date          TEXT NOT NULL,     -- snapshot date (semi-annual: YYYY-01-01 or YYYY-07-01)
    index_code    TEXT NOT NULL,     -- e.g. '000300' for CSI300
    symbol        TEXT NOT NULL,
    name          TEXT,
    sector        TEXT,              -- 东财行业分类
    PRIMARY KEY (date, index_code, symbol)
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

        # Migrate index_constituents: add date column, change PK
        ic_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(index_constituents)").fetchall()
        }
        if "updated_date" in ic_cols and "date" not in ic_cols:
            self._conn.executescript("""
                ALTER TABLE index_constituents RENAME TO index_constituents_old;
                CREATE TABLE index_constituents (
                    date       TEXT NOT NULL,
                    index_code TEXT NOT NULL,
                    symbol     TEXT NOT NULL,
                    name       TEXT,
                    sector     TEXT,
                    PRIMARY KEY (date, index_code, symbol)
                );
                INSERT INTO index_constituents (date, index_code, symbol, name, sector)
                SELECT updated_date, index_code, symbol, name, sector
                FROM index_constituents_old;
                DROP TABLE index_constituents_old;
            """)
            self._conn.commit()

        # Migrate fundamentals: add new columns if missing
        fund_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(fundamentals)").fetchall()
        }
        new_fund_cols = [
            "net_profit_margin", "gross_margin", "debt_ratio", "asset_turnover",
            "current_ratio", "equity_yoy", "asset_yoy", "cashflow_per_share",
            "cashflow_to_profit", "ps_ttm", "pcf_ttm", "pub_date",
            "roa", "operating_profit_ratio", "ocf_to_revenue",
            "eps_weighted", "quick_ratio", "receivables_turnover", "inventory_turnover",
            "cost_profit_ratio", "dividend_payout_ratio", "cash_ratio", "equity_ratio",
            "shareholder_equity_ratio", "undistributed_per_share", "capital_reserve_per_share",
        ]
        for col in new_fund_cols:
            if col not in fund_cols:
                col_type = "TEXT" if col == "pub_date" else "REAL"
                self._conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {col} {col_type}")
        self._conn.commit()

        # Migrate daily_bars: add new columns if missing
        bar_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(daily_bars)").fetchall()
        }
        new_bar_cols = [
            ("turn", "REAL"),
            ("preclose", "REAL"),
            ("tradestatus", "TEXT"),
            ("is_st", "INTEGER"),
        ]
        for col, col_type in new_bar_cols:
            if col not in bar_cols:
                self._conn.execute(f"ALTER TABLE daily_bars ADD COLUMN {col} {col_type}")
        # Remove ps_ttm/pcf_ttm from daily_bars (moved to fundamentals)
        if "ps_ttm" in bar_cols or "pcf_ttm" in bar_cols:
            # Migrate existing data: copy ps_ttm/pcf_ttm from daily_bars into fundamentals
            self._conn.execute("""
                INSERT OR REPLACE INTO fundamentals (symbol, date, pe_ttm, pb, ps_ttm, pcf_ttm,
                    roe_ttm, revenue_yoy, profit_yoy, market_cap)
                SELECT
                    b.symbol, b.date,
                    COALESCE(f.pe_ttm, NULL),
                    COALESCE(f.pb, NULL),
                    b.ps_ttm,
                    b.pcf_ttm,
                    COALESCE(f.roe_ttm, NULL),
                    COALESCE(f.revenue_yoy, NULL),
                    COALESCE(f.profit_yoy, NULL),
                    COALESCE(f.market_cap, NULL)
                FROM daily_bars b
                LEFT JOIN fundamentals f ON b.symbol = f.symbol AND b.date = f.date
                WHERE b.ps_ttm IS NOT NULL OR b.pcf_ttm IS NOT NULL
            """)
            for col in ("ps_ttm", "pcf_ttm"):
                if col in bar_cols:
                    self._conn.execute(f"ALTER TABLE daily_bars DROP COLUMN {col}")
        self._conn.commit()

    # ------------------------------------------------------------------
    # daily_bars
    # ------------------------------------------------------------------

    def save_daily_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE daily bars from *df*.

        *df* must contain columns: date, open, high, low, close, volume.
        Optional: amount, adj_factor, turn, preclose, tradestatus, is_st.
        Returns row count inserted.
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
                r.get("turn"),
                r.get("preclose"),
                r.get("tradestatus"),
                r.get("is_st"),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO daily_bars "
                "(symbol, date, open, high, low, close, volume, amount, adj_factor,"
                " turn, preclose, tradestatus, is_st) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def get_latest_bar_date(self) -> str | None:
        """Return the most recent date in daily_bars, or None."""
        with self._lock:
            row = self._conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
        return row[0] if row and row[0] else None

    def get_latest_news_timestamp(self) -> str | None:
        """Return the most recent news_items timestamp, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(timestamp) FROM news_items"
            ).fetchone()
        return row[0] if row and row[0] else None

    # ------------------------------------------------------------------
    # fundamentals
    # ------------------------------------------------------------------

    _FUND_COLS = [
        "pe_ttm", "pb", "roe_ttm", "revenue_yoy", "profit_yoy", "market_cap",
        "net_profit_margin", "gross_margin", "debt_ratio", "asset_turnover",
        "current_ratio", "equity_yoy", "asset_yoy", "cashflow_per_share",
        "cashflow_to_profit", "ps_ttm", "pcf_ttm", "pub_date",
        "roa", "operating_profit_ratio", "ocf_to_revenue",
        "eps_weighted", "quick_ratio", "receivables_turnover", "inventory_turnover",
        "cost_profit_ratio", "dividend_payout_ratio", "cash_ratio", "equity_ratio",
        "shareholder_equity_ratio", "undistributed_per_share", "capital_reserve_per_share",
    ]

    def save_fundamentals(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert fundamentals rows, merging non-NULL fields. Returns row count."""
        if df.empty:
            return 0
        cols = self._FUND_COLS
        rows: list[tuple[Any, ...]] = []
        for _, r in df.iterrows():
            rows.append(
                (symbol, str(r["date"])) + tuple(r.get(c) for c in cols)
            )
        col_list = ", ".join(["symbol", "date"] + cols)
        placeholders = ", ".join(["?"] * (2 + len(cols)))
        upsert = ", ".join(f"{c} = COALESCE(excluded.{c}, {c})" for c in cols)
        with self._lock:
            self._conn.executemany(
                f"INSERT INTO fundamentals ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT(symbol, date) DO UPDATE SET {upsert}",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def save_fundamentals_raw(self, symbol: str, df: pd.DataFrame) -> int:
        """Store full API response as JSON in ``fundamentals_raw`` table.

        Each row is keyed by (symbol, date) and stores the complete
        API response as a JSON string, preserving all 86+ columns for
        future use without re-running backfill.
        """
        import json
        import math

        if df.empty:
            return 0
        rows = []
        for _, r in df.iterrows():
            date_raw = r.get("日期")
            if date_raw is None or pd.isna(date_raw):
                continue
            date_val = str(date_raw)
            if not date_val:
                continue
            data = json.dumps(
                {k: v for k, v in r.items()
                 if pd.notna(v) and not (isinstance(v, float) and math.isinf(v))},
                ensure_ascii=False,
                default=str,
            )
            rows.append((symbol, date_val, data))
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT INTO fundamentals_raw (symbol, date, data) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol, date) DO UPDATE SET data = excluded.data",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def update_gross_margin_batch(self, quarter_date: str, data: dict[str, float]) -> int:
        """Batch update gross_margin for a given quarter date.

        Only updates rows that already exist in the fundamentals table —
        stocks outside the pool (no existing row) are silently skipped to
        avoid creating orphan rows.

        Returns number of rows affected.
        """
        if not data:
            return 0
        rows = [(val, sym, quarter_date) for sym, val in data.items()]
        with self._lock:
            cur = self._conn.executemany(
                "UPDATE fundamentals SET gross_margin = ? WHERE symbol = ? AND date = ?",
                rows,
            )
            self._conn.commit()
        return cur.rowcount

    def update_pub_date_batch(self, symbol: str, data: dict[str, str]) -> int:
        """Batch update pub_date for a given symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        data : dict[str, str]
            Mapping of ``report_date → pub_date`` (both YYYY-MM-DD).

        Returns number of rows affected.
        """
        if not data:
            return 0
        rows = [(pub, symbol, report) for report, pub in data.items()]
        with self._lock:
            cur = self._conn.executemany(
                "UPDATE fundamentals SET pub_date = ? WHERE symbol = ? AND date = ?",
                rows,
            )
            self._conn.commit()
        return cur.rowcount

    def update_pub_dates_by_quarter(self, quarter_date: str, data: dict[str, str]) -> int:
        """Batch update pub_date for all stocks in a given quarter.

        Parameters
        ----------
        quarter_date : str
            Quarter end date in ``YYYY-MM-DD`` format.
        data : dict[str, str]
            Mapping of ``symbol → pub_date``.

        Returns number of rows affected.
        """
        if not data:
            return 0
        rows = [(pub, sym, quarter_date) for sym, pub in data.items()]
        with self._lock:
            cur = self._conn.executemany(
                "UPDATE fundamentals SET pub_date = ? WHERE symbol = ? AND date = ?",
                rows,
            )
            self._conn.commit()
        return cur.rowcount

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

    # Columns that are quarterly (need ffill) vs daily (already complete)
    _QUARTERLY_COLS = [
        "roe_ttm", "revenue_yoy", "profit_yoy",
        "net_profit_margin", "gross_margin", "debt_ratio", "asset_turnover",
        "current_ratio", "equity_yoy", "asset_yoy", "cashflow_per_share",
        "cashflow_to_profit",
        "roa", "operating_profit_ratio", "ocf_to_revenue",
        "eps_weighted", "quick_ratio", "receivables_turnover", "inventory_turnover",
        "cost_profit_ratio", "dividend_payout_ratio", "cash_ratio", "equity_ratio",
        "shareholder_equity_ratio", "undistributed_per_share", "capital_reserve_per_share",
    ]

    def load_fundamentals_filled(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load fundamentals with quarterly columns forward-filled.

        PE/PB are daily (already complete). ROE, revenue_yoy, etc. are
        quarterly — only present on quarter-end dates. This method fetches
        the latest quarterly row before *start* as a seed for ffill, so
        the first trading day always has values.

        **PIT handling**: If a quarterly row has ``pub_date`` set and
        ``pub_date > date``, the quarterly values are delayed — they
        become effective only from ``pub_date`` onward, not from the
        report period end date.
        """
        with self._lock:
            # Find the latest quarterly row before start (seed for ffill)
            quarterly_check = " OR ".join(f"{c} IS NOT NULL" for c in self._QUARTERLY_COLS)
            seed_date = self._conn.execute(
                f"SELECT MAX(date) FROM fundamentals "
                f"WHERE symbol = ? AND date < ? AND ({quarterly_check})",
                (symbol, start),
            ).fetchone()[0]

            query_start = seed_date if seed_date else start
            df = pd.read_sql(
                "SELECT * FROM fundamentals WHERE symbol = ? AND date >= ? AND date <= ? "
                "ORDER BY date",
                self._conn,
                params=(symbol, query_start, end),
            )

        if df.empty:
            return df

        # PIT (Point-in-Time): delay quarterly data to announcement date.
        #
        # Without PIT, a Q1 report (period-end 2024-03-31, announced 2024-04-27)
        # would be ffilled starting from 03-31 — 27 days before anyone could
        # know the data. This loop moves each quarterly value from its
        # report-date row to the first row on or after pub_date.
        #
        # Multi-quarter example (same stock):
        #   Row 2023-12-31: Q4 roe=0.20, pub_date=2024-04-03
        #   Row 2024-03-31: Q1 roe=0.15, pub_date=2024-04-27
        #
        # After this loop:
        #   Row 2023-12-31: roe=NaN  (cleared — not yet announced)
        #   Row 2024-03-31: roe=NaN  (cleared — not yet announced)
        #   Row 2024-04-03: roe=0.20 (Q4 data lands here — first row ≥ pub_date)
        #   Row 2024-04-27: roe=0.15 (Q1 data lands here)
        #
        # After ffill:
        #   Dates before 2024-04-03 → NaN (or previous seed value)
        #   2024-04-03 to 2024-04-26 → Q4 value (0.20)
        #   2024-04-27 onward → Q1 value (0.15)
        #
        # Edge cases:
        # - pub_date falls on non-trading day → value lands on next available row
        # - pub_date beyond query range → value is dropped (not yet announced)
        # - pub_date is NULL (no backfill yet) → no delay, ffill from report date
        #   (backward compatible — behaves exactly as before PIT)
        has_pub = "pub_date" in df.columns
        if has_pub:
            for idx in df.index:
                pub = df.at[idx, "pub_date"]
                row_date = df.at[idx, "date"]
                if pd.notna(pub) and str(pub) > str(row_date):
                    target_mask = df["date"] >= str(pub)
                    if target_mask.any():
                        target_idx = df.index[target_mask][0]
                        for col in self._QUARTERLY_COLS:
                            if col not in df.columns:
                                continue
                            val = df.at[idx, col]
                            if pd.notna(val):
                                # Later (more recent) quarter overwrites earlier
                                # when both share the same pub_date
                                df.at[target_idx, col] = val
                                df.at[idx, col] = None

        for col in self._QUARTERLY_COLS:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Trim back to requested range (remove the seed row)
        if seed_date and seed_date < start:
            df = df[df["date"] >= start].reset_index(drop=True)
        return df

    def load_latest_fundamentals(self, symbols: list[str]) -> pd.DataFrame:
        """Load fundamentals with per-column latest non-NULL value for each symbol.

        PE/PB come from daily valuation rows while ROE/revenue/profit come from
        quarterly financial rows.  A simple ``MAX(date)`` would pick only the
        daily row and lose all quarterly fields.  This query coalesces the most
        recent non-NULL value for every column independently.
        """
        if not symbols:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        query = (
            f"SELECT symbol, "
            f"  MAX(date) AS date, "
            f"  (SELECT f2.pe_ttm FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.pe_ttm IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS pe_ttm, "
            f"  (SELECT f2.pb FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.pb IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS pb, "
            f"  (SELECT f2.roe_ttm FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.roe_ttm IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS roe_ttm, "
            f"  (SELECT f2.revenue_yoy FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.revenue_yoy IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS revenue_yoy, "
            f"  (SELECT f2.profit_yoy FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.profit_yoy IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS profit_yoy, "
            f"  (SELECT f2.market_cap FROM fundamentals f2 "
            f"   WHERE f2.symbol = f.symbol AND f2.market_cap IS NOT NULL "
            f"   ORDER BY f2.date DESC LIMIT 1) AS market_cap "
            f"FROM fundamentals f "
            f"WHERE symbol IN ({placeholders}) "
            f"GROUP BY symbol"
        )
        with self._lock:
            return pd.read_sql(query, self._conn, params=symbols)

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

        Each dict must have: date, symbol, index_code.
        Optional: name, sector.
        """
        if not rows:
            return 0
        tuples = [
            (r["date"], r["index_code"], r["symbol"],
             r.get("name"), r.get("sector"))
            for r in rows
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO index_constituents "
                "(date, index_code, symbol, name, sector) "
                "VALUES (?, ?, ?, ?, ?)",
                tuples,
            )
            self._conn.commit()
        return len(tuples)

    def delete_stale_index_constituents(self, active_codes: list[str]) -> int:
        """Delete constituents whose index_code is not in *active_codes*."""
        if not active_codes:
            return 0
        placeholders = ",".join("?" for _ in active_codes)
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM index_constituents WHERE index_code NOT IN ({placeholders})",
                active_codes,
            )
            self._conn.commit()
        return cur.rowcount

    def update_constituent_sectors(self, sector_map: dict[str, str]) -> int:
        """Update sector field for all rows matching each symbol.

        Parameters
        ----------
        sector_map : dict[str, str]
            Mapping of symbol → sector (e.g. ``{"600519": "白酒Ⅱ"}``).

        Returns the number of rows updated.
        """
        if not sector_map:
            return 0
        total = 0
        with self._lock:
            for symbol, sector in sector_map.items():
                cur = self._conn.execute(
                    "UPDATE index_constituents SET sector = ? "
                    "WHERE symbol = ? AND (sector IS NULL OR sector = '')",
                    (sector, symbol),
                )
                total += cur.rowcount
            self._conn.commit()
        return total

    def load_index_constituents(self, index_code: str, date: str | None = None) -> list[dict]:
        """Load constituents for a given index code.

        If *date* is given, return that snapshot. Otherwise return the latest snapshot.
        """
        with self._lock:
            if date:
                rows = self._conn.execute(
                    "SELECT date, index_code, symbol, name, sector "
                    "FROM index_constituents WHERE index_code = ? AND date = ? "
                    "ORDER BY symbol",
                    (index_code, date),
                ).fetchall()
            else:
                # Latest snapshot
                rows = self._conn.execute(
                    "SELECT date, index_code, symbol, name, sector "
                    "FROM index_constituents WHERE index_code = ? AND date = ("
                    "  SELECT MAX(date) FROM index_constituents WHERE index_code = ?"
                    ") ORDER BY symbol",
                    (index_code, index_code),
                ).fetchall()
        return [
            {"date": r[0], "index_code": r[1], "symbol": r[2],
             "name": r[3], "sector": r[4]}
            for r in rows
        ]

    def load_all_index_constituents(self, date: str | None = None) -> list[dict]:
        """Load constituents for all index codes.

        If *date* is given, return that snapshot. Otherwise return the latest snapshot.
        """
        with self._lock:
            if date:
                rows = self._conn.execute(
                    "SELECT date, index_code, symbol, name, sector "
                    "FROM index_constituents WHERE date = ? ORDER BY symbol",
                    (date,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT date, index_code, symbol, name, sector "
                    "FROM index_constituents WHERE date = ("
                    "  SELECT MAX(date) FROM index_constituents"
                    ") ORDER BY symbol",
                ).fetchall()
        return [
            {"date": r[0], "index_code": r[1], "symbol": r[2],
             "name": r[3], "sector": r[4]}
            for r in rows
        ]

    def load_constituents_for_date(self, target_date: str) -> list[str]:
        """Return stock symbols from the nearest snapshot <= target_date.

        Used for training/backtest to get the constituents active at a given point in time.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT symbol FROM index_constituents "
                "WHERE date = (SELECT MAX(date) FROM index_constituents WHERE date <= ?) "
                "ORDER BY symbol",
                (target_date,),
            ).fetchall()
        return [r[0] for r in rows]

    def load_constituents_union(self, start: str, end: str) -> list[str]:
        """Return all symbols that appeared in any snapshot between start and end.

        Used for training data: include all stocks that were ever in the index during the window.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT symbol FROM index_constituents "
                "WHERE date >= COALESCE("
                "  (SELECT MAX(date) FROM index_constituents WHERE date <= ?),"
                "  ?"
                ") AND date <= ? "
                "ORDER BY symbol",
                (start, start, end),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Adjustment factors
    # ------------------------------------------------------------------

    def get_daily_bar_dates(self, symbol: str) -> list[tuple[str, float]]:
        """Return all (date, adj_factor) pairs for a stock's daily bars."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT date, adj_factor FROM daily_bars "
                "WHERE symbol = ? ORDER BY date",
                (symbol,),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def update_adj_factors(
        self, symbol: str, updates: list[tuple[float, str]],
    ) -> None:
        """Batch update adj_factor for a stock's daily bars.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        updates : list of (new_factor, date)
            Each tuple is (adj_factor_value, date_string).
        """
        if not updates:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE daily_bars SET adj_factor = ? "
                "WHERE symbol = ? AND date = ?",
                [(factor, symbol, date) for factor, date in updates],
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
