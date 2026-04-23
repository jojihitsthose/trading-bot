"""
paper_db.py — SQLite database layer for the paper trading system.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", "paper_trades.db")

STARTING_BALANCE = 100_000.0


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            pairs         TEXT NOT NULL,
            scan_interval TEXT NOT NULL,
            risk_pct      REAL NOT NULL,
            rr            REAL NOT NULL,
            atr_sl_mult   REAL NOT NULL,
            loaded_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS open_positions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id   TEXT NOT NULL,
            instrument    TEXT NOT NULL,
            side          TEXT NOT NULL,
            entry_price   REAL NOT NULL,
            sl_price      REAL NOT NULL,
            tp_price      REAL NOT NULL,
            units         REAL NOT NULL,
            risk_usd      REAL NOT NULL,
            opened_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id   TEXT NOT NULL,
            instrument    TEXT NOT NULL,
            side          TEXT NOT NULL,
            entry_price   REAL NOT NULL,
            sl_price      REAL NOT NULL,
            tp_price      REAL NOT NULL,
            exit_price    REAL NOT NULL,
            exit_reason   TEXT NOT NULL,
            risk_usd      REAL NOT NULL,
            pnl           REAL NOT NULL,
            balance_after REAL NOT NULL,
            opened_at     TEXT NOT NULL,
            closed_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            balance     REAL NOT NULL,
            timestamp   TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def register_strategy(strategy_id, mod):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO strategies
            (strategy_id, name, pairs, scan_interval, risk_pct, rr, atr_sl_mult, loaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy_id,
        mod.NAME,
        json.dumps(mod.PAIRS),
        mod.SCAN_INTERVAL,
        mod.RISK_PCT,
        mod.RR,
        mod.ATR_SL_MULT,
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()
    conn.close()


def get_balance(strategy_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT SUM(pnl) as total_pnl FROM paper_trades WHERE strategy_id = ?",
        (strategy_id,)
    ).fetchone()
    conn.close()
    total_pnl = row["total_pnl"] or 0.0
    return STARTING_BALANCE + total_pnl


def open_position(strategy_id, instrument, side, entry_price, sl_price, tp_price, units, risk_usd):
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO open_positions
            (strategy_id, instrument, side, entry_price, sl_price, tp_price, units, risk_usd, opened_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy_id, instrument, side, entry_price,
        sl_price, tp_price, units, risk_usd,
        datetime.now(timezone.utc).isoformat(),
    ))
    pos_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pos_id


def close_position(pos_id, exit_price, exit_reason):
    """Move open position to paper_trades. Returns pnl."""
    conn = get_conn()
    pos = conn.execute(
        "SELECT * FROM open_positions WHERE id = ?", (pos_id,)
    ).fetchone()
    if not pos:
        conn.close()
        return 0.0

    rr      = conn.execute(
        "SELECT rr FROM strategies WHERE strategy_id = ?", (pos["strategy_id"],)
    ).fetchone()["rr"]

    risk_usd = pos["risk_usd"]
    if exit_reason == "TP":
        pnl = risk_usd * rr
    elif exit_reason == "SL":
        pnl = -risk_usd
    else:
        # Manual / timeout — calculate from price
        sl_dist = abs(pos["entry_price"] - pos["sl_price"])
        if pos["side"] == "BUY":
            move = exit_price - pos["entry_price"]
        else:
            move = pos["entry_price"] - exit_price
        pnl = risk_usd * (move / sl_dist) if sl_dist > 0 else 0.0

    balance_before = get_balance(pos["strategy_id"])
    balance_after  = balance_before + pnl

    conn.execute("""
        INSERT INTO paper_trades
            (strategy_id, instrument, side, entry_price, sl_price, tp_price,
             exit_price, exit_reason, risk_usd, pnl, balance_after, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pos["strategy_id"], pos["instrument"], pos["side"],
        pos["entry_price"], pos["sl_price"], pos["tp_price"],
        exit_price, exit_reason, risk_usd, pnl, balance_after,
        pos["opened_at"], datetime.now(timezone.utc).isoformat(),
    ))
    conn.execute("DELETE FROM open_positions WHERE id = ?", (pos_id,))
    conn.commit()
    conn.close()
    return pnl


def get_open_positions(strategy_id=None):
    conn = get_conn()
    if strategy_id:
        rows = conn.execute(
            "SELECT * FROM open_positions WHERE strategy_id = ? ORDER BY opened_at",
            (strategy_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM open_positions ORDER BY opened_at"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_history(strategy_id=None, limit=200):
    conn = get_conn()
    if strategy_id:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE strategy_id = ? ORDER BY closed_at DESC LIMIT ?",
            (strategy_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY closed_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(strategy_id):
    conn = get_conn()
    trades = conn.execute(
        "SELECT * FROM paper_trades WHERE strategy_id = ? ORDER BY closed_at",
        (strategy_id,)
    ).fetchall()
    conn.close()

    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "net_pct": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "balance": STARTING_BALANCE,
        }

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    balance = STARTING_BALANCE
    peak    = STARTING_BALANCE
    max_dd  = 0.0
    for t in trades:
        balance += t["pnl"]
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100
        max_dd   = max(max_dd, dd)

    net_pnl = balance - STARTING_BALANCE

    today_start = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "T00:00:00"
    today_pnl   = sum(t["pnl"] for t in trades if t["closed_at"] >= today_start)

    return {
        "total_trades":  len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      len(wins) / len(trades) * 100 if trades else 0.0,
        "net_pnl":       net_pnl,
        "net_pct":       net_pnl / STARTING_BALANCE * 100,
        "max_drawdown":  max_dd,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "avg_win":       gross_win / len(wins) if wins else 0.0,
        "avg_loss":      gross_loss / len(losses) if losses else 0.0,
        "balance":       balance,
        "today_pnl":     today_pnl,
    }


def save_equity_snapshot(strategy_id, balance):
    conn = get_conn()
    conn.execute(
        "INSERT INTO equity_snapshots (strategy_id, balance, timestamp) VALUES (?, ?, ?)",
        (strategy_id, balance, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def get_equity_snapshots(strategy_id, limit=500):
    conn = get_conn()
    rows = conn.execute("""
        SELECT timestamp, balance FROM equity_snapshots
        WHERE strategy_id = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (strategy_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def get_all_strategies():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM strategies ORDER BY loaded_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
