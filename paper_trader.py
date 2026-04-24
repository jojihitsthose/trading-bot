"""
paper_trader.py — 24/7 paper trading engine.

Loads strategies from the strategies/ folder automatically.
Simulates trades using real OANDA market data — no real orders placed.
Tracks every strategy independently in SQLite.

Usage:
    python paper_trader.py          # runs 24/7
    python paper_trader.py --once   # single scan then exit
"""

import os
import sys
import time
import glob
import logging
import argparse
import importlib.util
import threading
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import schedule
import requests as _requests

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as v20inst

import paper_db as db

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY          = os.environ.get("OANDA_API_KEY",    "a96040915699134d949c07f592b723c2-b883eecaeec7883a4c71b00c730b3593")
ENV              = os.environ.get("OANDA_ENV",        "practice")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "8295894132:AAGVD3E8L-YKsAgBZj9r_-O3gGvKb_huytI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "6756173967")
STRATEGIES_DIR   = os.path.join(os.path.dirname(__file__), "strategies")

PRICE_DEC = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "NZD_USD": 5, "USD_CAD": 5, "GBP_JPY": 3,
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("paper_trader.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── OANDA client ──────────────────────────────────────────────────────────────
client = API(access_token=API_KEY, environment=ENV)

# ── Strategy registry ─────────────────────────────────────────────────────────
STRATEGIES = {}   # strategy_id -> module

REQUIRED_ATTRS = ["NAME", "PAIRS", "SCAN_INTERVAL", "RISK_PCT", "RR", "ATR_SL_MULT", "get_signal"]


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        _requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# OANDA DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_candles(instrument: str, granularity: str, count: int) -> pd.DataFrame:
    params = {"granularity": granularity, "count": count + 1, "price": "M"}
    r = v20inst.InstrumentsCandles(instrument, params=params)
    client.request(r)
    rows = []
    for c in r.response["candles"]:
        if not c["complete"]:
            continue
        m = c["mid"]
        rows.append({
            "time":  pd.Timestamp(c["time"]).tz_localize(None),
            "open":  float(m["o"]),
            "high":  float(m["h"]),
            "low":   float(m["l"]),
            "close": float(m["c"]),
        })
    return pd.DataFrame(rows)


def get_mid_price(instrument: str) -> float:
    df = get_candles(instrument, "S5", 2)
    return float(df.iloc[-1]["close"]) if not df.empty else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_strategies():
    global STRATEGIES
    STRATEGIES = {}
    pattern = os.path.join(STRATEGIES_DIR, "*.py")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        strategy_id = name[:-3]
        try:
            spec = importlib.util.spec_from_file_location(strategy_id, path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            missing = [a for a in REQUIRED_ATTRS if not hasattr(mod, a)]
            if missing:
                log.warning(f"Strategy '{strategy_id}' missing attributes {missing} — skipped")
                continue

            STRATEGIES[strategy_id] = mod
            db.register_strategy(strategy_id, mod)
            log.info(f"Loaded strategy: {mod.NAME} ({strategy_id})  interval={mod.SCAN_INTERVAL}")

        except Exception as e:
            log.error(f"Failed to load strategy '{strategy_id}': {e}")

    log.info(f"Total strategies loaded: {len(STRATEGIES)}")


# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING  (mirrors live_trader.py compute_units)
# ══════════════════════════════════════════════════════════════════════════════

def compute_units(instrument: str, risk_usd: float, sl_dist: float) -> float:
    if sl_dist == 0:
        return 0.0
    if instrument.endswith("_USD"):
        return risk_usd / sl_dist
    elif instrument.startswith("USD_"):
        rate = get_mid_price(instrument)
        return risk_usd * rate / sl_dist
    else:
        quote = instrument.split("_")[1]
        rate  = get_mid_price(f"USD_{quote}")
        return risk_usd * rate / sl_dist


# ══════════════════════════════════════════════════════════════════════════════
# SL/TP MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def monitor_positions():
    """Check all open paper positions — close any that hit SL or TP."""
    positions = db.get_open_positions()
    if not positions:
        return

    # Group by instrument to minimise API calls
    instruments = list({p["instrument"] for p in positions})
    prices = {}
    for inst in instruments:
        try:
            prices[inst] = get_mid_price(inst)
        except Exception as e:
            log.warning(f"  [paper] Could not fetch price for {inst}: {e}")

    for pos in positions:
        inst  = pos["instrument"]
        price = prices.get(inst)
        if price is None:
            continue

        side     = pos["side"]
        hit_tp   = (side == "BUY"  and price >= pos["tp_price"]) or \
                   (side == "SELL" and price <= pos["tp_price"])
        hit_sl   = (side == "BUY"  and price <= pos["sl_price"]) or \
                   (side == "SELL" and price >= pos["sl_price"])

        if hit_tp or hit_sl:
            reason     = "TP" if hit_tp else "SL"
            exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
            pnl        = db.close_position(pos["id"], exit_price, reason)
            new_bal    = db.get_balance(pos["strategy_id"])
            strat_name = STRATEGIES[pos["strategy_id"]].NAME if pos["strategy_id"] in STRATEGIES else pos["strategy_id"]
            emoji      = "✅" if pnl >= 0 else "🔴"

            log.info(f"  [paper] {emoji} {reason}  {inst}  {side}  P&L ${pnl:+,.0f}  "
                     f"[{strat_name}]  bal=${new_bal:,.0f}")
            send_telegram(
                f"{emoji} PAPER TRADE CLOSED  [{strat_name}]\n"
                f"{side} {inst}\n"
                f"Entry: {pos['entry_price']}  |  Exit: {exit_price}\n"
                f"P&L: ${pnl:+,.2f}  |  Balance: ${new_bal:,.0f}\n"
                f"Result: {reason}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL SCAN
# ══════════════════════════════════════════════════════════════════════════════

_candle_cache: dict = {}


def _get_cached_candles(instrument, granularity, count):
    key = (instrument, granularity)
    if key not in _candle_cache:
        _candle_cache[key] = get_candles(instrument, granularity, count)
    return _candle_cache[key]


def scan_strategies(interval_filter: str):
    """
    Scan all strategies matching the given interval ('15min' or 'daily_09:02').
    Called from the scheduler.
    """
    global _candle_cache
    _candle_cache = {}   # flush per-cycle cache

    log.info("─" * 60)
    log.info(f"[paper] Scan ({interval_filter})  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    monitor_positions()

    for strategy_id, mod in STRATEGIES.items():
        if mod.SCAN_INTERVAL != interval_filter:
            continue

        open_pos = db.get_open_positions(strategy_id)
        open_insts = {p["instrument"] for p in open_pos}
        balance    = db.get_balance(strategy_id)

        for instrument in mod.PAIRS:
            if instrument in open_insts:
                continue

            try:
                h1 = _get_cached_candles(instrument, "H1", 350)
                h4 = _get_cached_candles(instrument, "H4", 120)
                d1 = _get_cached_candles(instrument, "D",  250)
                signal, atr = mod.get_signal(instrument, h1, h4, d1)
            except Exception as e:
                log.warning(f"  [paper] {mod.NAME} {instrument}: error — {e}")
                continue

            if signal is None:
                continue

            sl_dist  = atr * mod.ATR_SL_MULT
            tp_dist  = sl_dist * mod.RR
            risk_usd = balance * mod.RISK_PCT

            try:
                entry = get_mid_price(instrument)
            except Exception as e:
                log.warning(f"  [paper] Could not get entry price for {instrument}: {e}")
                continue

            dec = PRICE_DEC.get(instrument, 5)
            if signal == "BUY":
                sl_price = round(entry - sl_dist, dec)
                tp_price = round(entry + tp_dist, dec)
            else:
                sl_price = round(entry + sl_dist, dec)
                tp_price = round(entry - tp_dist, dec)

            units = compute_units(instrument, risk_usd, sl_dist)

            pos_id = db.open_position(
                strategy_id, instrument, signal,
                entry, sl_price, tp_price, units, risk_usd
            )
            open_insts.add(instrument)

            log.info(f"  [paper] OPENED  {signal} {instrument}  @ {entry:.{dec}f}  "
                     f"SL {sl_price:.{dec}f}  TP {tp_price:.{dec}f}  "
                     f"risk=${risk_usd:.0f}  [{mod.NAME}]")
            send_telegram(
                f"📝 PAPER TRADE OPENED  [{mod.NAME}]\n"
                f"{signal} {instrument}\n"
                f"Entry: {entry:.{dec}f}\n"
                f"SL: {sl_price:.{dec}f}  |  TP: {tp_price:.{dec}f}\n"
                f"Risk: ${risk_usd:.0f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# DAILY TELEGRAM SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def send_daily_summary():
    lines = [f"📊 PAPER TRADING DAILY SUMMARY\n{datetime.utcnow().strftime('%Y-%m-%d')} UTC\n"]
    for strategy_id, mod in STRATEGIES.items():
        stats   = db.get_stats(strategy_id)
        open_ps = db.get_open_positions(strategy_id)
        balance = db.get_balance(strategy_id)
        net_pct = (balance - db.STARTING_BALANCE) / db.STARTING_BALANCE * 100

        today_pnl = stats.get("today_pnl", 0.0)
        lines.append(
            f"{mod.NAME}\n"
            f"  Balance: ${balance:,.0f} ({net_pct:+.2f}%)\n"
            f"  Today: ${today_pnl:+,.0f}\n"
            f"  Win Rate (all-time): {stats['win_rate']:.1f}%  "
            f"| Trades: {stats['total_trades']}\n"
            f"  Open positions: {len(open_ps)}\n"
        )
    send_telegram("\n".join(lines))
    log.info("[paper] Daily summary sent")


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_equity_snapshots():
    for strategy_id in STRATEGIES:
        balance = db.get_balance(strategy_id)
        db.save_equity_snapshot(strategy_id, balance)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one scan cycle then exit")
    args = parser.parse_args()

    db.init_db()
    load_strategies()

    if not STRATEGIES:
        log.error("No strategies loaded — check the strategies/ folder")
        sys.exit(1)

    if args.once:
        scan_strategies("15min")
        scan_strategies("daily_09:02")
        return

    # Scheduler
    schedule.every(15).minutes.do(scan_strategies, interval_filter="15min")
    schedule.every().day.at("09:02").do(scan_strategies, interval_filter="daily_09:02")
    schedule.every().day.at("21:00").do(send_daily_summary)
    schedule.every().hour.do(save_equity_snapshots)

    log.info("Paper trader running — scanning every 15 min + daily at 09:02 UTC")
    log.info(f"Strategies: {', '.join(mod.NAME for mod in STRATEGIES.values())}")

    scan_strategies("15min")
    now_utc = datetime.now(timezone.utc)
    if now_utc.hour >= 9:
        scan_strategies("daily_09:02")
    save_equity_snapshots()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
