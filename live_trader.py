"""
live_trader.py — OANDA Demo/Live Execution Engine
MaxIncome strategy (Lever 3): 1.75% / 2.25% risk, max 2 simultaneous positions

SETUP:
    1. pip install oandapyV20 pandas numpy schedule
    2. Create free demo account at oanda.com
    3. Dashboard → Manage API Access → Generate Token
    4. Paste your API_KEY and ACCOUNT_ID in the CONFIG section below
    5. python live_trader.py

USAGE:
    python live_trader.py           # runs 24/7, wakes every hour at :02
    python live_trader.py --once    # single check then exit  (good for first test)
    python live_trader.py --dry     # signal check only, no orders placed

LOGS:
    Everything is written to live_trader.log and printed to terminal.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import schedule

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as v20inst
import oandapyV20.endpoints.orders     as v20orders
import oandapyV20.endpoints.trades     as v20trades
import oandapyV20.endpoints.accounts   as v20acct


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — fill in your API key and account ID here
# ══════════════════════════════════════════════════════════════════════════════
API_KEY    = os.environ.get("OANDA_API_KEY",    "a96040915699134d949c07f592b723c2-b883eecaeec7883a4c71b00c730b3593")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "101-001-39095268-001")
ENV        = "practice"   # "practice" = demo  |  "live" = real money

# ── Strategy parameters (MaxIncome Lever 3) ───────────────────────────────────
PAIRS = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD",
    "GBPJPY": "GBP_JPY",
    "NZDUSD": "NZD_USD",
}

RISK_WEAK      = 0.0175   # 1.75% of balance per trade
RISK_STRONG    = 0.0225   # 2.25% on ADX 36+ setups
ADX_WEAK       = 28
ADX_STRONG     = 36
RSI_OVERSOLD   = 40
RSI_OVERBOUGHT = 60
ATR_SL_MULT    = 2.0      # stop loss = entry ± ATR × 2.0
RR_HARD_TP     = 2.0      # take profit = entry ± SL_dist × 2.0
MAX_POSITIONS  = 3        # max simultaneous open positions
SESSION_START  = 7        # UTC — London open
SESSION_END    = 21       # UTC — NY close
MIN_ATR        = 0.0004   # skip if market too quiet (< 4 pips)
MAX_HOURS_HELD = 999      # no time-based force close

# Decimal places for price formatting (JPY pairs need 3, others need 5)
PRICE_DEC = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "NZD_USD": 5, "USD_CAD": 5, "GBP_JPY": 3,
}

# Correlation: never hold two of these in the same direction simultaneously
CORR_GROUPS = [
    ["EUR_USD", "GBP_USD", "NZD_USD", "AUD_USD"],
]


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("live_trader.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OANDA CLIENT
# ══════════════════════════════════════════════════════════════════════════════
client = None   # initialised in main() after config check


def _init_client():
    global client
    client = API(access_token=API_KEY, environment=ENV)


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_candles(instrument: str, granularity: str, count: int) -> pd.DataFrame:
    """Fetch completed OHLCV candles from OANDA."""
    params = {"granularity": granularity, "count": count + 1, "price": "M"}
    r = v20inst.InstrumentsCandles(instrument, params=params)
    client.request(r)
    rows = []
    for c in r.response["candles"]:
        if not c["complete"]:
            continue
        m = c["mid"]
        rows.append({
            "time":   pd.Timestamp(c["time"]).tz_localize(None),
            "open":   float(m["o"]),
            "high":   float(m["h"]),
            "low":    float(m["l"]),
            "close":  float(m["c"]),
            "volume": int(c["volume"]),
        })
    return pd.DataFrame(rows)


def get_balance() -> float:
    """Return current account balance in USD."""
    r = v20acct.AccountSummary(ACCOUNT_ID)
    client.request(r)
    return float(r.response["account"]["balance"])


def get_open_trades() -> list:
    """Return list of raw open trade dicts from OANDA."""
    r = v20trades.OpenTrades(ACCOUNT_ID)
    client.request(r)
    return r.response.get("trades", [])


def close_trade(trade_id: str, reason: str = ""):
    """Market-close a specific trade by ID."""
    try:
        r = v20trades.TradeClose(ACCOUNT_ID, trade_id)
        client.request(r)
        log.info(f"    Closed trade {trade_id}  [{reason}]")
    except V20Error as e:
        log.warning(f"    Failed to close trade {trade_id}: {e}")


def get_mid_price(instrument: str) -> float:
    """Fetch current mid price for an instrument."""
    df = get_candles(instrument, "S5", 2)
    return float(df.iloc[-1]["close"]) if not df.empty else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up  = df["high"].diff()
    dn  = -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _atr(df, period)
    pdi = 100 * pdm.ewm(com=period-1, min_periods=period).mean() / atr
    mdi = 100 * mdm.ewm(com=period-1, min_periods=period).mean() / atr
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(com=period-1, min_periods=period).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema8"]      = _ema(df["close"], 8)
    df["ema21"]     = _ema(df["close"], 21)
    df["ema50"]     = _ema(df["close"], 50)
    df["rsi"]       = _rsi(df["close"], 14)
    df["atr"]       = _atr(df, 14)
    df["adx"]       = _adx(df, 14)
    df["slope50"]   = df["ema50"].diff(3)
    df["atr_avg"]   = df["atr"].rolling(20).mean()
    df["atr_ratio"] = df["atr"] / df["atr_avg"].replace(0, np.nan)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def get_signal(instrument: str):
    """
    Fetch fresh candles and evaluate the last completed H1 bar.
    Returns ("BUY" | "SELL" | None, atr_value, risk_pct)
    """
    # Fetch data — need enough bars for all indicators
    h1 = get_candles(instrument, "H1",  350)   # EMA50 needs ~50, ADX ~14
    h4 = get_candles(instrument, "H4",  120)   # EMA50 on H4
    d1 = get_candles(instrument, "D",   250)   # EMA200 on D1

    if len(h1) < 60 or len(h4) < 60 or len(d1) < 210:
        log.debug(f"    {instrument}: not enough bars")
        return None, None, None

    h1 = compute_indicators(h1)
    curr = h1.iloc[-2]   # last COMPLETED bar
    prev = h1.iloc[-3]

    # Null guard
    for col in ["ema8", "ema21", "ema50", "rsi", "atr", "adx", "atr_ratio"]:
        if pd.isna(curr[col]):
            return None, None, None

    # Session filter (UTC)
    hour = curr["time"].hour
    if not (SESSION_START <= hour < SESSION_END):
        return None, None, None

    # ATR regime
    if curr["atr"] < MIN_ATR:
        return None, None, None
    if pd.isna(curr["atr_ratio"]) or not (0.5 <= curr["atr_ratio"] <= 2.5):
        return None, None, None

    # ADX
    if curr["adx"] < ADX_WEAK:
        return None, None, None

    # D1 macro trend (EMA200)
    d1["ema200"] = _ema(d1["close"], 200)
    d1_past = d1[d1["time"] <= curr["time"]]
    if d1_past.empty:
        return None, None, None
    d1_ema     = d1_past["ema200"].iloc[-1]
    macro_up   = curr["close"] > d1_ema
    macro_down = curr["close"] < d1_ema

    # H4 medium trend (EMA50)
    h4["ema50"] = _ema(h4["close"], 50)
    h4_past = h4[h4["time"] <= curr["time"]]
    if h4_past.empty:
        return None, None, None
    h4_ema  = h4_past["ema50"].iloc[-1]
    h4_up   = curr["close"] > h4_ema
    h4_down = curr["close"] < h4_ema

    risk_pct = RISK_STRONG if curr["adx"] >= ADX_STRONG else RISK_WEAK
    atr      = curr["atr"]

    # BUY: RSI recovered from oversold
    rsi_rec = prev["rsi"] < RSI_OVERSOLD and curr["rsi"] >= RSI_OVERSOLD
    if (macro_up and h4_up
            and curr["slope50"] > 0
            and curr["ema8"] > curr["ema21"]
            and rsi_rec):
        return "BUY", atr, risk_pct

    # SELL: RSI rejected from overbought
    rsi_rej = prev["rsi"] > RSI_OVERBOUGHT and curr["rsi"] <= RSI_OVERBOUGHT
    if (macro_down and h4_down
            and curr["slope50"] < 0
            and curr["ema8"] < curr["ema21"]
            and rsi_rej):
        return "SELL", atr, risk_pct

    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ══════════════════════════════════════════════════════════════════════════════

def compute_units(instrument: str, risk_usd: float, sl_dist: float) -> int:
    """
    Return OANDA units for the given USD risk and SL distance in price.

    P&L per unit (in quote currency) = price_move
    P&L in USD = P&L_in_quote × (USD per 1 quote unit)

    For XXX/USD (EURUSD, GBPUSD, AUDUSD, NZDUSD):
        quote is USD → units = risk_usd / sl_dist

    For USD/XXX (USDCAD):
        quote is CAD → risk_usd = units × sl_dist / USDCAD_rate
        → units = risk_usd × USDCAD_rate / sl_dist

    For cross pairs (GBPJPY):
        quote is JPY → risk_usd = units × sl_dist / USDJPY_rate
        → units = risk_usd × USDJPY_rate / sl_dist
    """
    if instrument.endswith("_USD"):
        # EUR_USD, GBP_USD, AUD_USD, NZD_USD
        units = risk_usd / sl_dist

    elif instrument.startswith("USD_"):
        # USD_CAD — quote = CAD, current USDCAD price converts CAD → USD
        usdcad = get_mid_price(instrument)
        units = risk_usd * usdcad / sl_dist

    else:
        # Cross pair: GBP_JPY — quote = JPY, need USDJPY rate
        quote = instrument.split("_")[1]           # "JPY"
        usd_quote = get_mid_price(f"USD_{quote}")  # USDJPY
        units = risk_usd * usd_quote / sl_dist

    return max(1, int(units))


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def correlation_blocked(instrument: str, signal: str, open_trades: list) -> bool:
    """
    Return True if placing this trade would create a correlated double position.
    E.g. already long EUR_USD → block long GBP_USD.
    """
    open_map = {}
    for t in open_trades:
        inst  = t["instrument"]
        side  = "BUY" if float(t["currentUnits"]) > 0 else "SELL"
        open_map[inst] = side

    for group in CORR_GROUPS:
        if instrument in group:
            for other in group:
                if other != instrument and other in open_map:
                    if open_map[other] == signal:
                        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ORDER PLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def place_order(instrument: str, signal: str, atr: float,
                risk_pct: float, balance: float, dry_run: bool = False):
    """Place a market order with attached SL and TP."""
    risk_usd = balance * risk_pct
    sl_dist  = atr * ATR_SL_MULT
    tp_dist  = sl_dist * RR_HARD_TP
    units    = compute_units(instrument, risk_usd, sl_dist)

    # Get current mid price for SL/TP calculation
    current = get_mid_price(instrument)
    dec     = PRICE_DEC.get(instrument, 5)

    if signal == "BUY":
        sl_price = round(current - sl_dist, dec)
        tp_price = round(current + tp_dist, dec)
        oanda_units = str(units)
    else:
        sl_price = round(current + sl_dist, dec)
        tp_price = round(current - tp_dist, dec)
        oanda_units = str(-units)

    log.info(f"  SIGNAL  {signal:4s}  {instrument:7s}  "
             f"@ {current:.{dec}f}  SL {sl_price:.{dec}f}  TP {tp_price:.{dec}f}  "
             f"units={units}  risk=${risk_usd:.0f}")

    if dry_run:
        log.info("    [DRY RUN — no order sent]")
        return

    order_data = {
        "order": {
            "type":        "MARKET",
            "instrument":  instrument,
            "units":       oanda_units,
            "timeInForce": "FOK",
            "stopLossOnFill": {
                "price": f"{sl_price:.{dec}f}",
            },
            "takeProfitOnFill": {
                "price": f"{tp_price:.{dec}f}",
            },
        }
    }

    try:
        r = v20orders.OrderCreate(ACCOUNT_ID, data=order_data)
        client.request(r)
        fill = r.response.get("orderFillTransaction", {})
        filled_price = fill.get("price", "?")
        trade_id     = fill.get("tradeOpened", {}).get("tradeID", "?")
        log.info(f"    Order filled  tradeID={trade_id}  filled@{filled_price}")
    except V20Error as e:
        log.error(f"    Order failed for {instrument}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT  (time exit only — SL/TP handled by OANDA)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_oanda_time(ts: str) -> datetime:
    """Parse OANDA timestamp safely — handles nanoseconds that Python 3.9 can't."""
    # Truncate to microseconds (6 decimal places) before parsing
    import re
    ts = ts.replace("Z", "+00:00")
    ts = re.sub(r'(\.\d{6})\d+', r'\1', ts)
    return datetime.fromisoformat(ts)


def manage_positions(dry_run: bool = False):
    """Force-close any trade that has been open longer than MAX_HOURS_HELD."""
    trades = get_open_trades()
    now    = datetime.now(timezone.utc)

    for t in trades:
        open_time  = _parse_oanda_time(t["openTime"])
        hours_open = (now - open_time).total_seconds() / 3600
        if hours_open >= MAX_HOURS_HELD:
            log.info(f"  TIME EXIT  tradeID={t['id']}  {t['instrument']}  "
                     f"open {hours_open:.1f}h")
            if not dry_run:
                close_trade(t["id"], reason=f"time exit ({hours_open:.0f}h)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIGNAL RUNNER  (called every hour)
# ══════════════════════════════════════════════════════════════════════════════

# Track open trade IDs to detect closures
_known_trade_ids: set = set()


def check_for_closed_trades(current_open: list):
    """Compare current open trades to last known set — log any that closed."""
    global _known_trade_ids
    current_ids = {t["id"] for t in current_open}

    closed_ids = _known_trade_ids - current_ids
    for tid in closed_ids:
        log.info(f"  TRADE CLOSED  tradeID={tid}  (hit SL, TP, or closed manually)")

    _known_trade_ids = current_ids


def run_signals(dry_run: bool = False):
    log.info("─" * 60)
    log.info(f"Signal check  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        balance      = get_balance()
        open_trades  = get_open_trades()
        n_open       = len(open_trades)

        # Detect any trades that closed since last check
        check_for_closed_trades(open_trades)

        # Catch-up: run strategies if they haven't fired today
        now_utc = datetime.now(timezone.utc)
        if now_utc.hour >= 7 and _fallback_last_date != _today_utc():
            run_fallback(dry_run=dry_run)
        if now_utc.hour >= 13 and _h1_strat_last_date != _today_utc():
            run_h1_strategy(dry_run=dry_run)

        log.info(f"  Balance: ${balance:,.2f}  |  Open trades: {n_open}/{MAX_POSITIONS}")

        # 1. Manage existing positions (time exits)
        manage_positions(dry_run=dry_run)
        # Re-fetch after potential closures
        open_trades = get_open_trades()
        n_open      = len(open_trades)

        # 2. Check each pair for a new signal
        for name, instrument in PAIRS.items():
            if n_open >= MAX_POSITIONS:
                log.info(f"  Max positions reached ({MAX_POSITIONS}) — skipping remaining pairs")
                break

            # Skip if we already have a position in this pair
            open_insts = {t["instrument"] for t in open_trades}
            if instrument in open_insts:
                log.debug(f"  {instrument}: already in trade")
                continue

            try:
                signal, atr, risk_pct = get_signal(instrument)
            except V20Error as e:
                log.warning(f"  {instrument}: API error — {e}")
                continue
            except Exception as e:
                log.warning(f"  {instrument}: unexpected error — {e}")
                continue

            if signal is None:
                log.debug(f"  {instrument}: no signal")
                continue

            # Correlation check
            if correlation_blocked(instrument, signal, open_trades):
                log.info(f"  {instrument}: {signal} blocked (correlation)")
                continue

            # Place the trade
            place_order(instrument, signal, atr, risk_pct, balance, dry_run=dry_run)
            n_open += 1

            # Re-fetch so correlation check has fresh data
            open_trades = get_open_trades()

    except V20Error as e:
        log.error(f"OANDA API error: {e}")
    except Exception as e:
        log.error(f"Unexpected error in run_signals: {e} — will retry next hour")


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK STRATEGY — London Open Momentum
# Fires once per day at 07:02 UTC if no trade has been placed today yet.
# Logic: trade in the direction of the H4 EMA21 trend, confirmed by D1 EMA200.
# Uses 1.0% risk (lower quality setup than the main strategy).
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_RISK    = 0.030   # 3.0% risk per trade
FALLBACK_PAIRS   = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD", "NZD_USD", "GBP_JPY"]

# Track the last date a fallback trade was placed (prevents multiple per day)
_fallback_last_date: str = ""


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _trade_placed_today(open_trades: list) -> bool:
    """Return True if any trade was opened today (UTC)."""
    today = _today_utc()
    for t in open_trades:
        open_time = _parse_oanda_time(t["openTime"])
        if open_time.strftime("%Y-%m-%d") == today:
            return True
    return False


def get_fallback_signal(instrument: str):
    """
    H4 trend-follow signal with quality filters.
    Returns ("BUY" | "SELL" | None, atr)

    Filters added vs original:
      - RSI must not be overbought (BUY) or oversold (SELL) on H4
      - Price must not be at a 20-bar high (BUY) or low (SELL) — avoids chasing
      - ADX > 20 — some trend must be present
    """
    h4 = get_candles(instrument, "H4", 80)
    d1 = get_candles(instrument, "D",  250)

    if len(h4) < 25 or len(d1) < 210:
        return None, None

    h4 = h4.copy()
    h4["ema21"] = _ema(h4["close"], 21)
    h4["atr"]   = _atr(h4, 14)
    h4["rsi"]   = _rsi(h4["close"], 14)
    h4["adx"]   = _adx(h4, 14)

    d1 = d1.copy()
    d1["ema200"] = _ema(d1["close"], 200)

    curr_h4 = h4.iloc[-2]   # last completed H4 bar
    d1_ema  = d1["ema200"].iloc[-1]

    if pd.isna(curr_h4["ema21"]) or pd.isna(curr_h4["atr"]) or pd.isna(d1_ema):
        return None, None
    if pd.isna(curr_h4["rsi"]) or pd.isna(curr_h4["adx"]):
        return None, None

    if curr_h4["atr"] < MIN_ATR:
        return None, None

    # ADX filter — skip ranging markets
    if curr_h4["adx"] < 20:
        return None, None

    price        = curr_h4["close"]
    recent_high  = h4["high"].iloc[-22:-2].max()   # 20-bar high (excluding current)
    recent_low   = h4["low"].iloc[-22:-2].min()
    rsi          = curr_h4["rsi"]

    # Both H4 EMA21 and D1 EMA200 must agree on direction
    if price > curr_h4["ema21"] and price > d1_ema:
        if rsi > 65:
            return None, None        # overbought — don't chase
        if price >= recent_high * 0.999:
            return None, None        # price at 20-bar high — don't chase
        return "BUY", curr_h4["atr"]

    if price < curr_h4["ema21"] and price < d1_ema:
        if rsi < 35:
            return None, None        # oversold — don't chase
        if price <= recent_low * 1.001:
            return None, None        # price at 20-bar low — don't chase
        return "SELL", curr_h4["atr"]

    return None, None


def run_fallback(dry_run: bool = False):
    """
    Fallback strategy: runs at 07:02 UTC daily.
    Places 1 trade on the strongest trending pair if no trade has been placed today.
    """
    global _fallback_last_date

    today = _today_utc()
    if _fallback_last_date == today:
        return   # already ran today

    log.info("─" * 60)
    log.info(f"Fallback check  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        balance     = get_balance()
        open_trades = get_open_trades()
        n_open      = len(open_trades)

        # Skip if main strategy already traded today or max positions reached
        if _trade_placed_today(open_trades):
            log.info("  Fallback: main strategy already traded today — skipping")
            _fallback_last_date = today
            return

        if n_open >= MAX_POSITIONS:
            log.info(f"  Fallback: max positions reached ({MAX_POSITIONS}) — skipping")
            _fallback_last_date = today
            return

        open_insts = {t["instrument"] for t in open_trades}

        placed = False
        for instrument in FALLBACK_PAIRS:
            if instrument in open_insts:
                continue
            if correlation_blocked(instrument, "BUY", open_trades):
                continue   # rough check — will be re-checked per signal below

            try:
                signal, atr = get_fallback_signal(instrument)
            except Exception as e:
                log.warning(f"  Fallback {instrument}: error — {e}")
                continue

            if signal is None:
                log.debug(f"  Fallback {instrument}: no signal")
                continue

            if correlation_blocked(instrument, signal, open_trades):
                log.info(f"  Fallback {instrument}: {signal} blocked (correlation)")
                continue

            log.info(f"  FALLBACK SIGNAL  {signal}  {instrument}  [1.0% risk]")
            place_order(instrument, signal, atr, FALLBACK_RISK, balance, dry_run=dry_run)
            placed = True
            break   # one trade per day only

        if not placed:
            log.info("  Fallback: no valid setup found on any pair today")

        _fallback_last_date = today

    except V20Error as e:
        log.error(f"Fallback OANDA API error: {e} — will retry next hour")
        # Don't set _fallback_last_date so it retries on next hourly check
    except Exception as e:
        log.error(f"Fallback unexpected error: {e} — will retry next hour")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — H1 Momentum (NY Open)
# Fires once per day at 13:02 UTC (NY open) if fewer than 2 positions are open.
# Uses H1 ATR for tighter SL/TP — faster resolution than Strategy 2.
# Direction confirmed by H4 EMA21 + D1 EMA200.
# ══════════════════════════════════════════════════════════════════════════════

_h1_strat_last_date: str = ""


def get_h1_signal(instrument: str):
    """
    H1-based momentum signal with H4 + D1 trend confirmation.
    Returns ("BUY" | "SELL" | None, atr)
    """
    h1 = get_candles(instrument, "H1", 100)
    h4 = get_candles(instrument, "H4", 60)
    d1 = get_candles(instrument, "D",  250)

    if len(h1) < 30 or len(h4) < 25 or len(d1) < 210:
        return None, None

    h1 = h1.copy()
    h1["ema21"] = _ema(h1["close"], 21)
    h1["atr"]   = _atr(h1, 14)
    h1["rsi"]   = _rsi(h1["close"], 14)
    h1["adx"]   = _adx(h1, 14)

    h4 = h4.copy()
    h4["ema21"] = _ema(h4["close"], 21)

    d1 = d1.copy()
    d1["ema200"] = _ema(d1["close"], 200)

    curr    = h1.iloc[-2]   # last completed H1 bar
    h4_ema  = h4["ema21"].iloc[-1]
    d1_ema  = d1["ema200"].iloc[-1]

    for val in [curr["ema21"], curr["atr"], curr["rsi"], curr["adx"], h4_ema, d1_ema]:
        if pd.isna(val):
            return None, None

    if curr["atr"] < MIN_ATR:
        return None, None
    if curr["adx"] < 20:
        return None, None

    price       = curr["close"]
    recent_high = h1["high"].iloc[-22:-2].max()
    recent_low  = h1["low"].iloc[-22:-2].min()
    rsi         = curr["rsi"]

    # All three timeframes must agree on direction
    if price > curr["ema21"] and price > h4_ema and price > d1_ema:
        if rsi > 65:
            return None, None
        if price >= recent_high * 0.999:
            return None, None
        return "BUY", curr["atr"]

    if price < curr["ema21"] and price < h4_ema and price < d1_ema:
        if rsi < 35:
            return None, None
        if price <= recent_low * 1.001:
            return None, None
        return "SELL", curr["atr"]

    return None, None


def run_h1_strategy(dry_run: bool = False):
    """
    Strategy 3: runs at 13:02 UTC daily (NY open).
    Places 1 trade if positions are available.
    """
    global _h1_strat_last_date

    today = _today_utc()
    if _h1_strat_last_date == today:
        return

    log.info("─" * 60)
    log.info(f"H1 Strategy check  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        balance     = get_balance()
        open_trades = get_open_trades()
        n_open      = len(open_trades)

        if n_open >= MAX_POSITIONS:
            log.info(f"  H1 Strategy: max positions reached ({MAX_POSITIONS}) — skipping")
            _h1_strat_last_date = today
            return

        open_insts = {t["instrument"] for t in open_trades}

        placed = False
        for instrument in FALLBACK_PAIRS:
            if instrument in open_insts:
                continue

            try:
                signal, atr = get_h1_signal(instrument)
            except Exception as e:
                log.warning(f"  H1 Strategy {instrument}: error — {e}")
                continue

            if signal is None:
                log.debug(f"  H1 Strategy {instrument}: no signal")
                continue

            if correlation_blocked(instrument, signal, open_trades):
                log.info(f"  H1 Strategy {instrument}: {signal} blocked (correlation)")
                continue

            log.info(f"  H1 STRATEGY SIGNAL  {signal}  {instrument}  [1.0% risk]")
            place_order(instrument, signal, atr, FALLBACK_RISK, balance, dry_run=dry_run)
            placed = True
            break

        if not placed:
            log.info("  H1 Strategy: no valid setup found today")

        _h1_strat_last_date = today

    except V20Error as e:
        log.error(f"H1 Strategy OANDA API error: {e} — will retry next hour")
    except Exception as e:
        log.error(f"H1 Strategy unexpected error: {e} — will retry next hour")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OANDA live trader")
    parser.add_argument("--once", action="store_true",
                        help="Run one signal check then exit")
    parser.add_argument("--dry",  action="store_true",
                        help="Signal check only — no orders placed")
    args = parser.parse_args()

    # Validate config
    if API_KEY == "YOUR_API_KEY_HERE" or ACCOUNT_ID == "YOUR_ACCOUNT_ID_HERE":
        print("\n[!] Edit live_trader.py and fill in your API_KEY and ACCOUNT_ID first.\n")
        sys.exit(1)

    _init_client()

    mode = "DRY RUN" if args.dry else ("LIVE" if ENV == "live" else "DEMO")
    log.info("=" * 60)
    log.info(f"  Live Trader starting  [{mode}]")
    log.info(f"  Account: {ACCOUNT_ID}  |  Pairs: {', '.join(PAIRS.keys())}")
    log.info(f"  Risk: {RISK_WEAK*100:.2f}% / {RISK_STRONG*100:.2f}%  |  "
             f"Max positions: {MAX_POSITIONS}")
    log.info("=" * 60)

    if args.once or args.dry:
        run_signals(dry_run=args.dry)
        return

    # Schedule: main strategy runs at :02 past every hour
    schedule.every().hour.at(":02").do(run_signals, dry_run=args.dry)

    # Fallback strategy: runs once per day at 07:02 UTC (London open)
    schedule.every().day.at("07:02").do(run_fallback, dry_run=args.dry)

    # Strategy 3 (H1 Momentum): runs once per day at 13:02 UTC (NY open)
    schedule.every().day.at("13:02").do(run_h1_strategy, dry_run=args.dry)

    log.info("Scheduler running — main strategy every hour at :02")
    log.info("                  — Strategy 2 (H4 fallback) daily at 07:02 UTC")
    log.info("                  — Strategy 3 (H1 momentum) daily at 13:02 UTC")
    log.info("Press Ctrl+C to stop\n")

    # Run immediately on start too
    run_signals(dry_run=args.dry)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
