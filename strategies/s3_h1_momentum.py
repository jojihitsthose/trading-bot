"""
Strategy S3 — H1 Momentum (London Open)
Fires once per day at 09:02 UTC. All three timeframes (H1/H4/D1) must agree on direction.
High conviction entries only — no overbought/oversold extremes, no resistance nearby.
"""
import pandas as pd
import numpy as np

NAME          = "S3 H1 Momentum"
PAIRS         = ["EUR_USD", "GBP_USD", "USD_CAD", "AUD_USD", "NZD_USD", "GBP_JPY"]
SCAN_INTERVAL = "daily_09:02"
RISK_PCT      = 0.050   # 5% risk per trade
RR            = 2.0
ATR_SL_MULT   = 1.2

MIN_ATR = 0.0004


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
    """
    h1 = candles_h1.copy()
    h4 = candles_h4.copy()
    d1 = candles_d1.copy()

    if len(h1) < 30 or len(h4) < 25 or len(d1) < 210:
        return None, None

    h1["ema21"] = _ema(h1["close"], 21)
    h1["atr"]   = _atr(h1, 14)
    h1["rsi"]   = _rsi(h1["close"], 14)
    h1["adx"]   = _adx(h1, 14)

    h4["ema21"]  = _ema(h4["close"], 21)
    d1["ema200"] = _ema(d1["close"], 200)

    curr   = h1.iloc[-2]
    h4_ema = h4["ema21"].iloc[-1]
    d1_ema = d1["ema200"].iloc[-1]

    for val in [curr["ema21"], curr["atr"], curr["rsi"], curr["adx"], h4_ema, d1_ema]:
        if pd.isna(val):
            return None, None

    if curr["atr"] < MIN_ATR or curr["adx"] < 20:
        return None, None

    price       = curr["close"]
    recent_high = h1["high"].iloc[-22:-2].max()
    recent_low  = h1["low"].iloc[-22:-2].min()
    rsi         = curr["rsi"]

    if price > curr["ema21"] and price > h4_ema and price > d1_ema:
        if rsi > 65 or price >= recent_high * 0.999:
            return None, None
        return "BUY", float(curr["atr"])

    if price < curr["ema21"] and price < h4_ema and price < d1_ema:
        if rsi < 35 or price <= recent_low * 1.001:
            return None, None
        return "SELL", float(curr["atr"])

    return None, None
