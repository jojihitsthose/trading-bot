"""
import_oanda_trades.py — One-time script to import all closed OANDA trades
into the paper trading dashboard database.

Run once:
    python3 import_oanda_trades.py
"""
from oandapyV20 import API
import oandapyV20.endpoints.trades as v20trades
import paper_db as db

API_KEY    = "a96040915699134d949c07f592b723c2-b883eecaeec7883a4c71b00c730b3593"
ACCOUNT_ID = "101-001-39095268-001"

client = API(access_token=API_KEY, environment="practice")

db.init_db()

# Fetch all closed trades
r = v20trades.TradesList(ACCOUNT_ID, params={"state": "CLOSED", "count": 500})
client.request(r)
closed = r.response.get("trades", [])
print(f"Found {len(closed)} closed trades on OANDA")

# Also fetch currently open trades
r2 = v20trades.OpenTrades(ACCOUNT_ID)
client.request(r2)
open_trades = r2.response.get("trades", [])
print(f"Found {len(open_trades)} currently open trades on OANDA")

PRICE_DEC = {
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "NZD_USD": 5, "USD_CAD": 5, "GBP_JPY": 3,
}

def strategy_for(units, pnl):
    """Guess strategy from trade size. S3 = 5% risk (~$4-6k loss), S1 = smaller."""
    abs_pnl = abs(float(pnl))
    if abs_pnl < 50:
        return None  # skip tiny test trades
    if abs_pnl >= 1500:
        return "s3_h1_momentum"
    return "s1_rsi_pullback"

imported = 0
skipped  = 0

conn = db.get_conn()

for t in closed:
    inst       = t["instrument"]
    units      = float(t["initialUnits"])
    entry_px   = float(t["price"])
    close_px   = float(t.get("averageClosePrice", t["price"]))
    pnl        = float(t.get("realizedPL", 0))
    opened_at  = t["openTime"][:26].replace("Z", "+00:00").replace("T", "T")
    closed_at  = t.get("closeTime", t["openTime"])[:26].replace("Z", "+00:00")
    side       = "BUY" if units > 0 else "SELL"

    strategy_id = strategy_for(units, pnl)
    if strategy_id is None:
        print(f"  SKIP  {inst}  P&L={pnl:.2f}  (test trade)")
        skipped += 1
        continue

    # Approximate risk_usd: for a loss it's abs(pnl), for a win it's pnl/2
    if pnl < 0:
        risk_usd = abs(pnl)
        exit_reason = "SL"
    else:
        risk_usd = pnl / 2.0
        exit_reason = "TP"

    # Approximate SL/TP from entry price and risk
    sl_dist = abs(entry_px - close_px) if pnl < 0 else abs(entry_px - close_px) / 2
    dec = PRICE_DEC.get(inst, 5)
    if side == "BUY":
        sl_price = round(entry_px - sl_dist, dec) if sl_dist > 0 else round(entry_px * 0.999, dec)
        tp_price = round(entry_px + sl_dist * 2, dec)
    else:
        sl_price = round(entry_px + sl_dist, dec) if sl_dist > 0 else round(entry_px * 1.001, dec)
        tp_price = round(entry_px - sl_dist * 2, dec)

    balance_before = db.get_balance(strategy_id)
    balance_after  = balance_before + pnl

    conn.execute("""
        INSERT INTO paper_trades
            (strategy_id, instrument, side, entry_price, sl_price, tp_price,
             exit_price, exit_reason, risk_usd, pnl, balance_after, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy_id, inst, side, entry_px, sl_price, tp_price,
        close_px, exit_reason, risk_usd, pnl, balance_after,
        opened_at, closed_at
    ))
    conn.commit()

    print(f"  IMPORTED  {side} {inst}  P&L=${pnl:+,.2f}  [{strategy_id}]")
    imported += 1

# Import open trades as open positions in the dashboard
for t in open_trades:
    inst     = t["instrument"]
    units    = float(t["currentUnits"])
    entry_px = float(t["price"])
    side     = "BUY" if units > 0 else "SELL"
    pnl_est  = float(t.get("unrealizedPL", 0))

    # Determine strategy
    abs_units = abs(units)
    strategy_id = "s3_h1_momentum"  # open trades are all S3

    # Check not already in open_positions
    existing = conn.execute(
        "SELECT id FROM open_positions WHERE strategy_id=? AND instrument=?",
        (strategy_id, inst)
    ).fetchone()
    if existing:
        print(f"  SKIP OPEN  {inst} (already in DB)")
        continue

    dec = PRICE_DEC.get(inst, 5)
    balance = db.get_balance(strategy_id)
    risk_usd = balance * 0.05  # S3 = 5% risk

    # Get SL/TP from OANDA trade details
    sl_price = float(t.get("stopLossOrder", {}).get("price", 0)) or \
               (round(entry_px * 0.997, dec) if side == "BUY" else round(entry_px * 1.003, dec))
    tp_price = float(t.get("takeProfitOrder", {}).get("price", 0)) or \
               (round(entry_px * 1.006, dec) if side == "BUY" else round(entry_px * 0.994, dec))

    conn.execute("""
        INSERT INTO open_positions
            (strategy_id, instrument, side, entry_price, sl_price, tp_price,
             units, risk_usd, opened_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy_id, inst, side, entry_px, sl_price, tp_price,
        abs(units), risk_usd, t["openTime"][:26].replace("Z", "+00:00")
    ))
    conn.commit()
    print(f"  IMPORTED OPEN  {side} {inst}  unreal=${pnl_est:+,.2f}  [{strategy_id}]")

conn.close()
print(f"\nDone — imported {imported} closed trades, skipped {skipped} test trades")
