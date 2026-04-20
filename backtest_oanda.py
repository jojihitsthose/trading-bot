"""
backtest_oanda.py — 1-month backtest for all 3 strategies using OANDA data.

Strategy 1: Main RSI Pullback   — H1 signal, H1/H4/D1 confirmation, fires every hour
Strategy 2: H4 Fallback         — H4 trend-follow, fires once/day at London open (07:00 UTC)
Strategy 3: H1 Momentum         — H1 signal, H4/D1 confirmation, fires once/day at NY open (13:00 UTC)

Usage:
    python3 backtest_oanda.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from oandapyV20 import API
import oandapyV20.endpoints.instruments as v20inst

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OANDA_API_KEY",    "a96040915699134d949c07f592b723c2-b883eecaeec7883a4c71b00c730b3593")
ENV     = "practice"

PAIRS_S1 = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD",
    "GBPJPY": "GBP_JPY",
    "NZDUSD": "NZD_USD",
}
PAIRS_S2_S3 = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD", "NZD_USD", "GBP_JPY"]

ADX_WEAK       = 28
RSI_OVERSOLD   = 40
RSI_OVERBOUGHT = 60
ATR_SL_MULT    = 2.0
RR_HARD_TP     = 2.0
SESSION_START  = 7
SESSION_END    = 21
MIN_ATR        = 0.0004
MAX_BARS_HELD  = 72   # backtest time-stop (3 days on H1, 12 days on H4)

STARTING_BALANCE = 98_946.0   # USD — actual account balance
RISK_S1          = 0.0175     # 1.75% per S1 trade
RISK_S2_S3       = 0.020      # 2.0%  per S3 trade

client = API(access_token=API_KEY, environment=ENV)


# ── Data fetch ────────────────────────────────────────────────────────────────

def get_candles(instrument, granularity, count):
    params = {"granularity": granularity, "count": count, "price": "M"}
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


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def _adx(df, period=14):
    up  = df["high"].diff()
    dn  = -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _atr(df, period)
    pdi = 100 * pdm.ewm(com=period-1, min_periods=period).mean() / atr
    mdi = 100 * mdm.ewm(com=period-1, min_periods=period).mean() / atr
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(com=period-1, min_periods=period).mean()


# ── Simulate trade outcome bar-by-bar ─────────────────────────────────────────

def simulate_trade(bars_df, entry_idx, side, sl, tp, entry_price):
    """
    Walk forward from entry_idx+1.
    Returns (result, bars_held, exit_time, exit_price)
    result = 'WIN' | 'LOSS' | 'TIMEOUT'
    """
    for j in range(1, MAX_BARS_HELD + 1):
        k = entry_idx + j
        if k >= len(bars_df):
            last = bars_df.iloc[-1]
            return "TIMEOUT", j, last["time"], last["close"]
        bar = bars_df.iloc[k]
        if side == "BUY":
            if bar["low"] <= sl:
                return "LOSS", j, bar["time"], sl
            if bar["high"] >= tp:
                return "WIN", j, bar["time"], tp
        else:
            if bar["high"] >= sl:
                return "LOSS", j, bar["time"], sl
            if bar["low"] <= tp:
                return "WIN", j, bar["time"], tp
    last_k = min(entry_idx + MAX_BARS_HELD, len(bars_df) - 1)
    last   = bars_df.iloc[last_k]
    return "TIMEOUT", MAX_BARS_HELD, last["time"], last["close"]


def calc_pnl(risk_usd, result, entry_price, exit_price, side, sl_dist):
    """
    WIN   → +risk * RR
    LOSS  → -risk
    TIMEOUT → actual price move scaled to risk
    """
    if result == "WIN":
        return risk_usd * RR_HARD_TP
    if result == "LOSS":
        return -risk_usd
    # TIMEOUT: scale actual price move against the SL distance
    if sl_dist == 0:
        return 0.0
    move = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
    return risk_usd * (move / sl_dist)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — Main RSI Pullback (H1, all hours)
# ══════════════════════════════════════════════════════════════════════════════

def s1_signal(h1, h4, d1, i):
    if i < 2:
        return None
    curr = h1.iloc[i]
    prev = h1.iloc[i - 1]

    for col in ["ema8", "ema21", "ema50", "rsi", "atr", "adx", "atr_ratio"]:
        if pd.isna(curr[col]):
            return None

    hour = curr["time"].hour
    if not (SESSION_START <= hour < SESSION_END):
        return None
    if curr["atr"] < MIN_ATR:
        return None
    if pd.isna(curr["atr_ratio"]) or not (0.5 <= curr["atr_ratio"] <= 2.5):
        return None
    if curr["adx"] < ADX_WEAK:
        return None

    d1_past = d1[d1["time"] <= curr["time"]]
    if d1_past.empty or pd.isna(d1_past["ema200"].iloc[-1]):
        return None
    d1_ema = d1_past["ema200"].iloc[-1]

    h4_past = h4[h4["time"] <= curr["time"]]
    if h4_past.empty or pd.isna(h4_past["ema50"].iloc[-1]):
        return None
    h4_ema = h4_past["ema50"].iloc[-1]

    rsi_rec = prev["rsi"] < RSI_OVERSOLD  and curr["rsi"] >= RSI_OVERSOLD
    rsi_rej = prev["rsi"] > RSI_OVERBOUGHT and curr["rsi"] <= RSI_OVERBOUGHT

    if (curr["close"] > d1_ema and curr["close"] > h4_ema
            and curr["slope50"] > 0 and curr["ema8"] > curr["ema21"] and rsi_rec):
        return "BUY"
    if (curr["close"] < d1_ema and curr["close"] < h4_ema
            and curr["slope50"] < 0 and curr["ema8"] < curr["ema21"] and rsi_rej):
        return "SELL"
    return None


def backtest_s1_pair(name, instrument):
    print(f"    {name}...", end=" ", flush=True)
    h1 = get_candles(instrument, "H1", 5000)
    h4 = get_candles(instrument, "H4", 2000)
    d1 = get_candles(instrument, "D",  500)

    if len(h1) < 300:
        print("not enough data")
        return []

    h1 = h1.copy()
    h1["ema8"]      = _ema(h1["close"], 8)
    h1["ema21"]     = _ema(h1["close"], 21)
    h1["ema50"]     = _ema(h1["close"], 50)
    h1["rsi"]       = _rsi(h1["close"], 14)
    h1["atr"]       = _atr(h1, 14)
    h1["adx"]       = _adx(h1, 14)
    h1["slope50"]   = h1["ema50"].diff(3)
    h1["atr_avg"]   = h1["atr"].rolling(20).mean()
    h1["atr_ratio"] = h1["atr"] / h1["atr_avg"].replace(0, np.nan)

    h4["ema50"]  = _ema(h4["close"], 50)
    d1["ema200"] = _ema(d1["close"], 200)

    trades   = []
    in_trade = False
    exit_bar = -1

    for i in range(60, len(h1) - 1):
        if in_trade and i <= exit_bar:
            continue
        in_trade = False

        sig = s1_signal(h1, h4, d1, i)
        if not sig:
            continue

        entry = h1.iloc[i + 1]["open"]
        atr   = h1.iloc[i]["atr"]
        sl_d  = atr * ATR_SL_MULT
        tp_d  = sl_d * RR_HARD_TP
        sl    = entry - sl_d if sig == "BUY" else entry + sl_d
        tp    = entry + tp_d if sig == "BUY" else entry - tp_d

        result, bars, exit_time, exit_price = simulate_trade(h1, i + 1, sig, sl, tp, entry)
        exit_bar = i + 1 + bars
        in_trade = True

        trades.append({
            "strategy":    "S1_Main",
            "pair":        name,
            "side":        sig,
            "entry_time":  h1.iloc[i]["time"],
            "exit_time":   exit_time,
            "result":      result,
            "bars_held":   bars,
            "entry_price": entry,
            "exit_price":  exit_price,
            "sl_dist":     sl_d,
            "risk_pct":    RISK_S1,
        })

    print(f"{len(trades)} trades")
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — H4 Fallback (once per day at London open ~07:00 UTC)
# ══════════════════════════════════════════════════════════════════════════════

def s2_signal(h4_row, h4_df, d1_ema, row_idx):
    curr = h4_row
    if pd.isna(curr["ema21"]) or pd.isna(curr["atr"]) or pd.isna(curr["rsi"]) or pd.isna(curr["adx"]):
        return None, None
    if curr["atr"] < MIN_ATR:
        return None, None
    if curr["adx"] < 25:
        return None, None

    price       = curr["close"]
    start       = max(0, row_idx - 21)
    recent_high = h4_df["high"].iloc[start:row_idx].max()
    recent_low  = h4_df["low"].iloc[start:row_idx].min()
    rsi         = curr["rsi"]

    if price > curr["ema21"] and price > d1_ema:
        if rsi > 65:
            return None, None
        if price >= recent_high * 0.999:
            return None, None
        return "BUY", curr["atr"]

    if price < curr["ema21"] and price < d1_ema:
        if rsi < 35:
            return None, None
        if price <= recent_low * 1.001:
            return None, None
        return "SELL", curr["atr"]

    return None, None


def backtest_s2_pair(instrument):
    name = instrument.replace("_", "")
    h4 = get_candles(instrument, "H4", 2000)
    d1 = get_candles(instrument, "D",  500)

    if len(h4) < 25 or len(d1) < 210:
        return []

    h4 = h4.copy()
    h4["ema21"] = _ema(h4["close"], 21)
    h4["atr"]   = _atr(h4, 14)
    h4["rsi"]   = _rsi(h4["close"], 14)
    h4["adx"]   = _adx(h4, 14)

    d1 = d1.copy()
    d1["ema200"] = _ema(d1["close"], 200)

    trades    = []
    last_date = None
    in_trade  = False
    exit_bar  = -1

    for i in range(25, len(h4) - 1):
        if in_trade and i <= exit_bar:
            continue
        in_trade = False

        bar_time = h4.iloc[i]["time"]
        # Strategy 2 only fires once per day at the first H4 bar at/after 07:00 UTC
        if bar_time.hour < 7:
            continue
        bar_date = bar_time.date()
        if bar_date == last_date:
            continue

        d1_past = d1[d1["time"] <= bar_time]
        if d1_past.empty or pd.isna(d1_past["ema200"].iloc[-1]):
            continue
        d1_ema = d1_past["ema200"].iloc[-1]

        sig, atr = s2_signal(h4.iloc[i], h4, d1_ema, i)
        last_date = bar_date  # mark day as checked regardless

        if not sig:
            continue

        entry = h4.iloc[i + 1]["open"]
        sl_d  = atr * ATR_SL_MULT
        tp_d  = sl_d * RR_HARD_TP
        sl    = entry - sl_d if sig == "BUY" else entry + sl_d
        tp    = entry + tp_d if sig == "BUY" else entry - tp_d

        result, bars, exit_time, exit_price = simulate_trade(h4, i + 1, sig, sl, tp, entry)
        exit_bar = i + 1 + bars
        in_trade = True

        trades.append({
            "strategy":    "S2_Fallback",
            "pair":        name,
            "side":        sig,
            "entry_time":  h4.iloc[i]["time"],
            "exit_time":   exit_time,
            "result":      result,
            "bars_held":   bars,
            "entry_price": entry,
            "exit_price":  exit_price,
            "sl_dist":     sl_d,
            "risk_pct":    RISK_S2_S3,
        })

    return trades


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — H1 Momentum (once per day at NY open ~13:00 UTC)
# ══════════════════════════════════════════════════════════════════════════════

def s3_signal(h1_row, h1_df, h4_ema, d1_ema, row_idx):
    curr = h1_row
    if pd.isna(curr["ema21"]) or pd.isna(curr["atr"]) or pd.isna(curr["rsi"]) or pd.isna(curr["adx"]):
        return None, None
    if curr["atr"] < MIN_ATR:
        return None, None
    if curr["adx"] < 20:
        return None, None

    price       = curr["close"]
    start       = max(0, row_idx - 21)
    recent_high = h1_df["high"].iloc[start:row_idx].max()
    recent_low  = h1_df["low"].iloc[start:row_idx].min()
    rsi         = curr["rsi"]

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


def backtest_s3_pair(instrument):
    name = instrument.replace("_", "")
    h1 = get_candles(instrument, "H1", 5000)
    h4 = get_candles(instrument, "H4", 2000)
    d1 = get_candles(instrument, "D",  500)

    if len(h1) < 30 or len(h4) < 25 or len(d1) < 210:
        return []

    h1 = h1.copy()
    h1["ema21"] = _ema(h1["close"], 21)
    h1["atr"]   = _atr(h1, 14)
    h1["rsi"]   = _rsi(h1["close"], 14)
    h1["adx"]   = _adx(h1, 14)

    h4 = h4.copy()
    h4["ema21"] = _ema(h4["close"], 21)

    d1 = d1.copy()
    d1["ema200"] = _ema(d1["close"], 200)

    trades    = []
    last_date = None
    in_trade  = False
    exit_bar  = -1

    for i in range(25, len(h1) - 1):
        if in_trade and i <= exit_bar:
            continue
        in_trade = False

        bar_time = h1.iloc[i]["time"]
        # Strategy 3 only fires once per day at the first H1 bar at/after 13:00 UTC
        if bar_time.hour < 13:
            continue
        bar_date = bar_time.date()
        if bar_date == last_date:
            continue

        h4_past = h4[h4["time"] <= bar_time]
        d1_past = d1[d1["time"] <= bar_time]
        if h4_past.empty or d1_past.empty:
            continue
        if pd.isna(h4_past["ema21"].iloc[-1]) or pd.isna(d1_past["ema200"].iloc[-1]):
            continue

        h4_ema = h4_past["ema21"].iloc[-1]
        d1_ema = d1_past["ema200"].iloc[-1]

        sig, atr = s3_signal(h1.iloc[i], h1, h4_ema, d1_ema, i)
        last_date = bar_date  # mark day as checked

        if not sig:
            continue

        entry = h1.iloc[i + 1]["open"]
        sl_d  = atr * ATR_SL_MULT
        tp_d  = sl_d * RR_HARD_TP
        sl    = entry - sl_d if sig == "BUY" else entry + sl_d
        tp    = entry + tp_d if sig == "BUY" else entry - tp_d

        result, bars, exit_time, exit_price = simulate_trade(h1, i + 1, sig, sl, tp, entry)
        exit_bar = i + 1 + bars
        in_trade = True

        trades.append({
            "strategy":    "S3_H1Mom",
            "pair":        name,
            "side":        sig,
            "entry_time":  h1.iloc[i]["time"],
            "exit_time":   exit_time,
            "result":      result,
            "bars_held":   bars,
            "entry_price": entry,
            "exit_price":  exit_price,
            "sl_dist":     sl_d,
            "risk_pct":    RISK_S2_S3,
        })

    return trades


# ── Print helpers ─────────────────────────────────────────────────────────────

def add_pnl(df, start_balance):
    """Add risk_usd and pnl columns; return final balance and max drawdown."""
    df = df.sort_values("entry_time").copy()
    balance  = start_balance
    peak     = start_balance
    max_dd   = 0.0
    pnl_list = []
    bal_list = []

    for _, row in df.iterrows():
        risk_usd = balance * row["risk_pct"]
        pnl      = calc_pnl(risk_usd, row["result"], row["entry_price"],
                            row["exit_price"], row["side"], row["sl_dist"])
        balance += pnl
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100
        max_dd   = max(max_dd, dd)
        pnl_list.append(pnl)
        bal_list.append(balance)

    df["pnl"]     = pnl_list
    df["balance"] = bal_list
    return df, balance, max_dd


def print_strategy_summary(label, trades_df, all_months, start_balance):
    if trades_df.empty:
        print(f"  {label}: no trades")
        return

    trades_df, final_bal, max_dd = add_pnl(trades_df, start_balance)

    wins    = len(trades_df[trades_df["result"] == "WIN"])
    losses  = len(trades_df[trades_df["result"] == "LOSS"])
    timeout = len(trades_df[trades_df["result"] == "TIMEOUT"])
    total   = len(trades_df)
    wr      = wins / total * 100 if total > 0 else 0
    months  = trades_df["month"].nunique()
    tpm     = total / months if months > 0 else 0
    net     = final_bal - start_balance
    net_pct = net / start_balance * 100

    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Trades: {total}  ({wins}W / {losses}L / {timeout} timeout)  "
          f"Win rate: {wr:.0f}%  Avg/month: {tpm:.1f}")
    print(f"  Net P&L: ${net:+,.0f}  ({net_pct:+.1f}%)  |  "
          f"Max drawdown: {max_dd:.1f}%  |  Final balance: ${final_bal:,.0f}")
    print()

    print(f"  {'Month':<10} {'Trades':>7} {'Win%':>6} {'P&L':>10} {'Balance':>10}")
    running_bal = start_balance
    for m in sorted(all_months):
        mt  = trades_df[trades_df["month"] == m]
        if mt.empty:
            continue
        mw    = mt[mt["result"] == "WIN"]
        mwr   = len(mw) / len(mt) * 100 if len(mt) > 0 else 0
        mpnl  = mt["pnl"].sum()
        running_bal += mpnl
        print(f"  {str(m):<10} {len(mt):>7} {mwr:>5.0f}%  "
              f"${mpnl:>+8,.0f}  ${running_bal:>9,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BACKTEST — All 3 Strategies  (~7 months OANDA data)")
    print("=" * 60)

    all_trades = []

    # ── Strategy 1: Main RSI Pullback ─────────────────────────────────────────
    print("\n[Strategy 1 — Main RSI Pullback, H1 signal, 6 pairs]")
    for name, instrument in PAIRS_S1.items():
        trades = backtest_s1_pair(name, instrument)
        all_trades.extend(trades)

    # ── Strategy 3: H1 Momentum ───────────────────────────────────────────────
    print("\n[Strategy 3 — H1 Momentum, NY open, 3 pairs]")
    for instrument in PAIRS_S2_S3:
        name = instrument.replace("_", "")
        print(f"    {name}...", end=" ", flush=True)
        trades = backtest_s3_pair(instrument)
        print(f"{len(trades)} trades")
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades generated.")
        return

    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"]      = df["entry_time"].dt.to_period("M")
    all_months       = sorted(df["month"].unique())

    # ── Per-strategy summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULTS BY STRATEGY  (starting balance: ${STARTING_BALANCE:,.0f})")
    print("=" * 60)

    label_map = {
        "S1_Main":  "Strategy 1 — Main RSI Pullback (H1, 6 pairs)",
        "S3_H1Mom": "Strategy 3 — H1 Momentum     (H1, 6 pairs, 13:00 UTC)",
    }
    for strat in ["S1_Main", "S3_H1Mom"]:
        print_strategy_summary(label_map[strat], df[df["strategy"] == strat].copy(),
                               all_months, STARTING_BALANCE)

    # ── Combined monthly totals ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  COMBINED — all 3 strategies, starting ${STARTING_BALANCE:,.0f}")
    print("=" * 60)

    df_all, final_bal, max_dd = add_pnl(df.copy(), STARTING_BALANCE)
    wins    = len(df_all[df_all["result"] == "WIN"])
    losses  = len(df_all[df_all["result"] == "LOSS"])
    timeout = len(df_all[df_all["result"] == "TIMEOUT"])
    total   = len(df_all)
    net     = final_bal - STARTING_BALANCE
    net_pct = net / STARTING_BALANCE * 100

    print(f"  Trades: {total}  ({wins}W / {losses}L / {timeout} timeout)  "
          f"Win rate: {wins/total*100:.0f}%  Avg/month: {total/len(all_months):.1f}")
    print(f"  Net P&L: ${net:+,.0f}  ({net_pct:+.1f}%)  |  "
          f"Max drawdown: {max_dd:.1f}%  |  Final balance: ${final_bal:,.0f}")
    print()
    print(f"  {'Month':<10} {'Trades':>7} {'Win%':>6} {'P&L':>10} {'Balance':>10}")
    running_bal = STARTING_BALANCE
    for m in all_months:
        mt   = df_all[df_all["month"] == m]
        if mt.empty:
            continue
        mw   = mt[mt["result"] == "WIN"]
        mwr  = len(mw) / len(mt) * 100 if len(mt) > 0 else 0
        mpnl = mt["pnl"].sum()
        running_bal += mpnl
        print(f"  {str(m):<10} {len(mt):>7} {mwr:>5.0f}%  "
              f"${mpnl:>+8,.0f}  ${running_bal:>9,.0f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
