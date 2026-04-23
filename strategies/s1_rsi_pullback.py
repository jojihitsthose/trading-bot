"""
Strategy S1 — RSI Pullback
Scans every 15 minutes. Requires RSI to recover from oversold (or reject from overbought)
with H4 + D1 trend confirmation and ADX trend filter.
"""
import pandas as pd
import numpy as np

NAME          = "S1 RSI Pullback"
PAIRS         = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CAD", "GBP_JPY", "NZD_USD"]
SCAN_INTERVAL = "15min"
RISK_PCT      = 0.0175   # 1.75% (weak ADX) — conservative for S1
RR            = 2.0
ATR_SL_MULT   = 1.2

# Thresholds
ADX_WEAK       = 28
RSI_OVERSOLD   = 40
RSI_OVERBOUGHT = 60
MIN_ATR        = 0.0004
SESSION_START  = 7
SESSION_END    = 21


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

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


def get_signal(instrument, candles_h1, candles_h4, candles_d1):
    """
    Returns ("BUY"|"SELL", atr) or (None, None).
    candles_h1/h4/d1 are pre-fetched DataFrames passed by the paper trader engine.
    """
    h1 = candles_h1.copy()
    h4 = candles_h4.copy()
    d1 = candles_d1.copy()

    if len(h1) < 60 or len(h4) < 60 or len(d1) < 210:
        return None, None

    h1["ema8"]      = _ema(h1["close"], 8)
    h1["ema21"]     = _ema(h1["close"], 21)
    h1["ema50"]     = _ema(h1["close"], 50)
    h1["rsi"]       = _rsi(h1["close"], 14)
    h1["atr"]       = _atr(h1, 14)
    h1["adx"]       = _adx(h1, 14)
    h1["slope50"]   = h1["ema50"].diff(3)
    h1["atr_avg"]   = h1["atr"].rolling(20).mean()
    h1["atr_ratio"] = h1["atr"] / h1["atr_avg"].replace(0, np.nan)

    curr = h1.iloc[-2]
    prev = h1.iloc[-3]

    for col in ["ema8", "ema21", "ema50", "rsi", "atr", "adx", "atr_ratio"]:
        if pd.isna(curr[col]):
            return None, None

    hour = curr["time"].hour
    if not (SESSION_START <= hour < SESSION_END):
        return None, None
    if curr["atr"] < MIN_ATR:
        return None, None
    if not (0.5 <= curr["atr_ratio"] <= 2.5):
        return None, None
    if curr["adx"] < ADX_WEAK:
        return None, None

    d1["ema200"] = _ema(d1["close"], 200)
    d1_past = d1[d1["time"] <= curr["time"]]
    if d1_past.empty or pd.isna(d1_past["ema200"].iloc[-1]):
        return None, None
    d1_ema = d1_past["ema200"].iloc[-1]

    h4["ema50"] = _ema(h4["close"], 50)
    h4_past = h4[h4["time"] <= curr["time"]]
    if h4_past.empty or pd.isna(h4_past["ema50"].iloc[-1]):
        return None, None
    h4_ema = h4_past["ema50"].iloc[-1]

    rsi_rec = prev["rsi"] < RSI_OVERSOLD  and curr["rsi"] >= RSI_OVERSOLD
    rsi_rej = prev["rsi"] > RSI_OVERBOUGHT and curr["rsi"] <= RSI_OVERBOUGHT

    if (curr["close"] > d1_ema and curr["close"] > h4_ema
            and curr["slope50"] > 0 and curr["ema8"] > curr["ema21"] and rsi_rec):
        return "BUY", float(curr["atr"])

    if (curr["close"] < d1_ema and curr["close"] < h4_ema
            and curr["slope50"] < 0 and curr["ema8"] < curr["ema21"] and rsi_rej):
        return "SELL", float(curr["atr"])

    return None, None
