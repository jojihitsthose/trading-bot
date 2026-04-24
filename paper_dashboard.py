"""
paper_dashboard.py — Live web dashboard for the paper trading system.
Auto-refreshes every 30 seconds. Run alongside paper_trader.py.

Usage:
    python paper_dashboard.py
    Open http://localhost:5051 in your browser.
    On Railway, the PORT env var is used automatically.
"""

import os
import json
from datetime import datetime, timezone

from flask import Flask, jsonify
import requests as _requests

from oandapyV20 import API
import oandapyV20.endpoints.instruments as v20inst

import paper_db as db

API_KEY = os.environ.get("OANDA_API_KEY", "a96040915699134d949c07f592b723c2-b883eecaeec7883a4c71b00c730b3593")
ENV     = os.environ.get("OANDA_ENV",     "practice")
PORT    = int(os.environ.get("PORT", 5051))

client = API(access_token=API_KEY, environment=ENV)
app    = Flask(__name__)

COLORS = ["#00d4aa", "#ff6b6b", "#ffd93d", "#a29bfe", "#fd79a8", "#74b9ff", "#55efc4"]


def _get_mid_price(instrument):
    try:
        params = {"granularity": "S5", "count": 2, "price": "M"}
        r = v20inst.InstrumentsCandles(instrument, params=params)
        client.request(r)
        candles = [c for c in r.response["candles"] if c["complete"]]
        if candles:
            return float(candles[-1]["mid"]["c"])
    except Exception:
        pass
    return None


def _render_dashboard():
    strategies = db.get_all_strategies()
    all_stats  = {}
    all_equity = {}
    all_open   = {}

    for s in strategies:
        sid             = s["strategy_id"]
        all_stats[sid]  = db.get_stats(sid)
        all_equity[sid] = db.get_equity_snapshots(sid, limit=200)
        all_open[sid]   = db.get_open_positions(sid)

    trades = db.get_trade_history(limit=100)

    # Fetch current prices for open positions (deduplicated)
    open_positions_all = db.get_open_positions()
    price_map = {}
    for p in open_positions_all:
        inst = p["instrument"]
        if inst not in price_map:
            price_map[inst] = _get_mid_price(inst)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; font-size: 1.6rem; }}
  .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }}
  h2 {{ color: #8b949e; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin: 28px 0 12px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; min-width: 220px; flex: 1; }}
  .card-name {{ font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; }}
  .card-balance {{ font-size: 1.5rem; font-weight: 700; }}
  .card-pct {{ font-size: 0.9rem; margin-bottom: 10px; }}
  .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
  .card-row {{ display: flex; justify-content: space-between; font-size: 0.82rem; color: #8b949e; margin-top: 4px; }}
  .card-row span {{ color: #e6edf3; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 8px 10px; border-bottom: 1px solid #30363d; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #161b22; }}
  .win {{ color: #3fb950; }} .loss {{ color: #f85149; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 8px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; }}
  .badge-buy {{ background: #1a3a2e; color: #3fb950; }}
  .badge-sell {{ background: #3a1a1a; color: #f85149; }}
</style>
</head>
<body>
<h1>Paper Trading Dashboard</h1>
<div class="subtitle">Last updated: {now_utc} &nbsp;|&nbsp; Auto-refreshes every 30s</div>
"""

    # ── Strategy Cards ────────────────────────────────────────────────────────
    html += '<h2>Strategy Overview</h2><div class="cards">'
    for i, s in enumerate(strategies):
        sid   = s["strategy_id"]
        stats = all_stats[sid]
        bal   = stats["balance"]
        pct   = stats["net_pct"]
        pnl   = stats["net_pnl"]
        today = stats.get("today_pnl", 0.0)
        color = COLORS[i % len(COLORS)]
        pct_cls  = "pos" if pct  >= 0 else "neg"
        today_cls = "pos" if today >= 0 else "neg"

        html += f"""
<div class="card" style="border-top: 3px solid {color}">
  <div class="card-name">{s['name']}</div>
  <div class="card-balance">${bal:,.0f}</div>
  <div class="card-pct {pct_cls}">{pct:+.2f}% &nbsp; (${pnl:+,.0f})</div>
  <div class="card-row">Win Rate <span>{stats['win_rate']:.1f}%</span></div>
  <div class="card-row">Profit Factor <span>{stats['profit_factor']:.2f}</span></div>
  <div class="card-row">Total Trades <span>{stats['total_trades']}</span></div>
  <div class="card-row">Max Drawdown <span>{stats['max_drawdown']:.1f}%</span></div>
  <div class="card-row">Today P&amp;L <span class="{today_cls}">${today:+,.0f}</span></div>
  <div class="card-row">Open Positions <span>{len(all_open[sid])}</span></div>
</div>"""
    html += '</div>'

    # ── Equity Curves ─────────────────────────────────────────────────────────
    html += '<h2>Equity Curves</h2><div class="chart-wrap"><canvas id="equityChart" height="80"></canvas></div>'

    datasets = []
    for i, s in enumerate(strategies):
        sid      = s["strategy_id"]
        snaps    = all_equity[sid]
        color    = COLORS[i % len(COLORS)]
        labels   = [sn["timestamp"][:16].replace("T", " ") for sn in snaps]
        balances = [sn["balance"] for sn in snaps]
        datasets.append({
            "label": s["name"],
            "data": [{"x": l, "y": b} for l, b in zip(labels, balances)],
            "borderColor": color,
            "backgroundColor": color + "22",
            "borderWidth": 2,
            "pointRadius": 0,
            "tension": 0.3,
            "fill": False,
        })

    html += f"""
<script>
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{ datasets: {json.dumps(datasets)} }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#8b949e' }} }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    scales: {{
      x: {{ type: 'category', ticks: {{ color: '#8b949e', maxTicksLimit: 10 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});
</script>"""

    # ── Open Positions ────────────────────────────────────────────────────────
    html += '<h2>Open Positions</h2>'
    if open_positions_all:
        html += """<table>
<tr><th>Strategy</th><th>Pair</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th>
<th>Current</th><th>Unreal P&amp;L</th><th>Opened</th></tr>"""
        strat_names = {s["strategy_id"]: s["name"] for s in strategies}
        for p in open_positions_all:
            inst    = p["instrument"]
            dec     = 3 if "JPY" in inst else 5
            price   = price_map.get(inst)
            sid     = p["strategy_id"]
            rr      = db.get_conn().execute("SELECT rr FROM strategies WHERE strategy_id=?", (sid,)).fetchone()
            rr_val  = rr["rr"] if rr else 2.0
            db.get_conn().close()

            if price:
                sl_dist   = abs(p["entry_price"] - p["sl_price"])
                if p["side"] == "BUY":
                    move = price - p["entry_price"]
                else:
                    move = p["entry_price"] - price
                unreal = p["risk_usd"] * (move / sl_dist) if sl_dist > 0 else 0.0
                unreal_str = f'<span class="{"pos" if unreal >= 0 else "neg"}">${unreal:+,.0f}</span>'
                price_str  = f"{price:.{dec}f}"
            else:
                unreal_str = "—"
                price_str  = "—"

            badge = f'<span class="badge badge-{"buy" if p["side"]=="BUY" else "sell"}">{p["side"]}</span>'
            html += f"""<tr>
<td>{strat_names.get(p['strategy_id'], p['strategy_id'])}</td>
<td>{inst}</td><td>{badge}</td>
<td>{p['entry_price']:.{dec}f}</td>
<td>{p['sl_price']:.{dec}f}</td>
<td>{p['tp_price']:.{dec}f}</td>
<td>{price_str}</td>
<td>{unreal_str}</td>
<td>{p['opened_at'][:16].replace('T',' ')} UTC</td>
</tr>"""
        html += "</table>"
    else:
        html += '<p style="color:#8b949e; padding: 8px 0">No open positions</p>'

    # ── Stats Comparison ──────────────────────────────────────────────────────
    html += '<h2>Stats Comparison</h2><table><tr><th>Metric</th>'
    for s in strategies:
        html += f"<th>{s['name']}</th>"
    html += "</tr>"

    metrics = [
        ("Total Trades",   lambda st: str(st["total_trades"])),
        ("Wins / Losses",  lambda st: f"{st['wins']}W / {st['losses']}L"),
        ("Win Rate",       lambda st: f"{st['win_rate']:.1f}%"),
        ("Profit Factor",  lambda st: f"{st['profit_factor']:.2f}"),
        ("Net P&L",        lambda st: f"${st['net_pnl']:+,.0f}"),
        ("Total Return",   lambda st: f"{st['net_pct']:+.2f}%"),
        ("Max Drawdown",   lambda st: f"{st['max_drawdown']:.1f}%"),
        ("Avg Win",        lambda st: f"${st['avg_win']:,.0f}"),
        ("Avg Loss",       lambda st: f"${st['avg_loss']:,.0f}"),
    ]
    for label, fn in metrics:
        html += f"<tr><td>{label}</td>"
        for s in strategies:
            html += f"<td>{fn(all_stats[s['strategy_id']])}</td>"
        html += "</tr>"
    html += "</table>"

    # ── Recent Trade Log ──────────────────────────────────────────────────────
    html += '<h2>Recent Trades (last 100)</h2>'
    strat_names = {s["strategy_id"]: s["name"] for s in strategies}
    if trades:
        html += """<table>
<tr><th>Closed</th><th>Strategy</th><th>Pair</th><th>Side</th>
<th>Entry</th><th>Exit</th><th>P&amp;L</th><th>Balance</th><th>Result</th></tr>"""
        for t in trades:
            inst = t["instrument"]
            dec  = 3 if "JPY" in inst else 5
            cls  = "win" if t["pnl"] >= 0 else "loss"
            badge = f'<span class="badge badge-{"buy" if t["side"]=="BUY" else "sell"}">{t["side"]}</span>'
            html += f"""<tr>
<td>{t['closed_at'][:16].replace('T',' ')} UTC</td>
<td>{strat_names.get(t['strategy_id'], t['strategy_id'])}</td>
<td>{inst}</td><td>{badge}</td>
<td>{t['entry_price']:.{dec}f}</td>
<td>{t['exit_price']:.{dec}f}</td>
<td class="{cls}">${t['pnl']:+,.2f}</td>
<td>${t['balance_after']:,.0f}</td>
<td class="{cls}">{t['exit_reason']}</td>
</tr>"""
        html += "</table>"
    else:
        html += '<p style="color:#8b949e; padding: 8px 0">No completed trades yet</p>'

    html += "</body></html>"
    return html


@app.route("/")
def index():
    return _render_dashboard()


@app.route("/api/data")
def api_data():
    strategies = db.get_all_strategies()
    result = {}
    for s in strategies:
        sid = s["strategy_id"]
        result[sid] = {
            "info":    s,
            "stats":   db.get_stats(sid),
            "open":    db.get_open_positions(sid),
            "equity":  db.get_equity_snapshots(sid, limit=100),
            "trades":  db.get_trade_history(sid, limit=50),
        }
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    import threading
    db.init_db()

    # Start the paper trader scanner in a background thread
    def start_scanner():
        import paper_trader
        paper_trader.db.init_db()
        paper_trader.load_strategies()
        if paper_trader.STRATEGIES:
            import schedule, time
            from datetime import datetime, timezone
            schedule.every(15).minutes.do(paper_trader.scan_strategies, interval_filter="15min")
            schedule.every().day.at("09:02").do(paper_trader.scan_strategies, interval_filter="daily_09:02")
            schedule.every().day.at("21:00").do(paper_trader.send_daily_summary)
            schedule.every().hour.do(paper_trader.save_equity_snapshots)
            paper_trader.scan_strategies("15min")
            if datetime.now(timezone.utc).hour >= 9:
                paper_trader.scan_strategies("daily_09:02")
            paper_trader.save_equity_snapshots()
            while True:
                schedule.run_pending()
                time.sleep(30)

    scanner_thread = threading.Thread(target=start_scanner, daemon=True)
    scanner_thread.start()

    print(f"Paper trading system starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
