#!/usr/bin/env python3
"""Plot recent Cloud Hunter trades with Ichimoku cloud"""
import json, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2
CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load_last_month(sym):
    p = os.path.join('/data/trading28/data/whale_15m_1y', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    ts = j.get('ts', [])
    if not ts: return None
    c = np.array(j['c'], float); h = np.array(j['h'], float)
    l = np.array(j['l'], float); o = np.array(j['o'], float)
    mask = np.array(ts) >= CUTOFF
    if mask.sum() < 200: return None
    return (c[mask], h[mask], l[mask], o[mask], [t for i,t in enumerate(ts) if mask[i]])

def resample_8h(c, h, l, o, ts):
    idx = pd.to_datetime(np.array(ts), unit='ms')
    df = pd.DataFrame({'o':o,'h':h,'l':l,'c':c}, index=idx)
    r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index

def get_entries(c, h, l, o):
    tenkan, kijun, senkou = 3, 9, 18; n = len(c)
    if n < senkou + 30: return np.zeros(n, bool), None, None
    h_t = pd.Series(h).rolling(tenkan).max().values
    l_t = pd.Series(l).rolling(tenkan).min().values
    t_arr = (h_t + l_t) / 2
    h_k = pd.Series(h).rolling(kijun).max().values
    l_k = pd.Series(l).rolling(kijun).min().values
    k_arr = (h_k + l_k) / 2
    h_s = pd.Series(h).rolling(senkou).max().values
    l_s = pd.Series(l).rolling(senkou).min().values
    sb_raw = (h_s + l_s) / 2; sa_raw = (t_arr + k_arr) / 2
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n: sa[i+shift] = sa_raw[i]; sb[i+shift] = sb_raw[i]
    
    entries = np.zeros(n, bool)
    cloud_top = np.maximum(sa, sb)
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]): continue
        if c[i] > cloud_top[i] and t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]:
            entries[i] = True
    return entries, sa, sb, t_arr, k_arr

def run_trades(entries, c, h, l, o, idx):
    n = len(c); trades = []; pos = 0; ep = 0; cool = 0; entry_idx = None
    for i in range(n):
        exit_signal = False
        if pos:
            if h[i] >= ep * 1.05: pnl = 5 - COMM*100; exit_signal = True
            elif l[i] <= ep * 0.975:
                pnl = max((c[i]/ep-1)*100 - COMM*100, -2.5*MAX_SLIPPAGE-COMM*100)
                exit_signal = True
            if exit_signal:
                trades.append({'entry_time': entry_idx, 'exit_time': idx[i], 'entry_bar': entry_bar, 'exit_bar': i, 'ep': ep, 'exit_p': c[i], 'pnl': pnl})
                pos = 0; cool = COOLDOWN
        if not pos and cool == 0 and entries[i]:
            pos = 1; ep = c[i]; entry_idx = idx[i]; entry_bar = i
        if not pos and cool > 0: cool -= 1
    if pos:
        trades.append({'entry_time': entry_idx, 'exit_time': idx[-1], 'entry_bar': entry_bar, 'exit_bar': n-1, 'ep': ep, 'exit_p': c[-1], 'pnl': (c[-1]/ep-1)*100-COMM*100})
    return trades

def plot_trade(sym, c, h, l, o, idx, sa, sb, t_arr, k_arr, trade, filename):
    """Plot one trade with Ichimoku cloud context"""
    # Show 30 bars before entry, entire trade, 10 bars after exit
    start = max(0, trade['entry_bar'] - 30)
    end = min(len(c), trade['exit_bar'] + 15)
    
    c_slice = c[start:end]; h_slice = h[start:end]
    l_slice = l[start:end]; o_slice = o[start:end]
    idx_slice = idx[start:end]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Candlesticks (simplified)
    colors = ['#26a69a' if c_slice[i] >= o_slice[i] else '#ef5350' for i in range(len(c_slice))]
    body_width = 0.4
    for i in range(len(c_slice)):
        ax.plot([i, i], [l_slice[i], h_slice[i]], color=colors[i], linewidth=1)
        ax.add_patch(plt.Rectangle((i-body_width/2, min(o_slice[i], c_slice[i])), body_width, abs(c_slice[i]-o_slice[i]), 
                                    facecolor=colors[i], edgecolor=colors[i], alpha=0.9))
    
    # Ichimoku cloud
    if sa is not None:
        sa_slice = sa[start:end]; sb_slice = sb[start:end]
        x_range = np.arange(len(c_slice))
        ax.fill_between(x_range, sa_slice, sb_slice, where=~np.isnan(sa_slice), alpha=0.2, color='#4CAF50', label='Cloud')
    
    # Tenkan & Kijun
    if t_arr is not None:
        t_slice = t_arr[start:end]; k_slice = k_arr[start:end]
        ax.plot(x_range, t_slice, color='#2196F3', linewidth=1, alpha=0.7, label='Tenkan (3)')
        ax.plot(x_range, k_slice, color='#FF9800', linewidth=1, alpha=0.7, label='Kijun (9)')
    
    # Entry marker
    entry_x = trade['entry_bar'] - start
    ax.scatter(entry_x, trade['ep'], color='lime', s=150, zorder=5, marker='^', edgecolors='darkgreen', linewidths=1.5, label='ENTRY')
    
    # Exit marker
    exit_x = trade['exit_bar'] - start
    color = 'lime' if trade['pnl'] > 0 else 'red'
    marker = 'v' if trade['pnl'] > 0 else 'v'
    ax.scatter(exit_x, trade['exit_p'], color=color, s=150, zorder=5, marker=marker, edgecolors='black', linewidths=1.5, label=f"EXIT ({trade['pnl']:+.1f}%)")
    
    # TP/SL lines from entry
    if trade['exit_bar'] >= trade['entry_bar']:
        tp_line = trade['ep'] * 1.05; sl_line = trade['ep'] * 0.975
        ax.axhline(y=tp_line, xmin=entry_x/len(c_slice), xmax=min(exit_x/len(c_slice), 1), color='green', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axhline(y=sl_line, xmin=entry_x/len(c_slice), xmax=min(exit_x/len(c_slice), 1), color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    
    # Date labels
    date_labels = [idx_slice[i].strftime('%m/%d %H:%M') if i % 5 == 0 else '' for i in range(len(idx_slice))]
    ax.set_xticks(range(len(c_slice)))
    ax.set_xticklabels(date_labels, rotation=45, fontsize=7, ha='right')
    
    date_range = f"{idx_slice[0].strftime('%m/%d')} → {idx_slice[-1].strftime('%m/%d')}"
    ax.set_title(f'{sym} | Cloud Hunter | {date_range} | PnL: {trade["pnl"]:+.1f}%', fontweight='bold')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return filename

# Main
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Find coins with good recent trades
sym_trades = {}
for sym in ['STRK', 'QKC', 'GLM', 'NMR', 'AWE', 'REQ', 'XEC', 'HBAR', 'ZEN', 'GRT', 'STX', 'THETA', 'DOT', 'UNI', 'ATOM', 'DYDX']:
    data = load_last_month(sym)
    if data is None: continue
    c, h, l, o, ts = data
    resampled = resample_8h(c, h, l, o, ts)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    entries, sa, sb, t_arr, k_arr = get_entries(c8, h8, l8, o8)
    trades = run_trades(entries, c8, h8, l8, o8, idx)
    if trades:
        sym_trades[sym] = (c8, h8, l8, o8, idx, sa, sb, t_arr, k_arr, trades)

# Plot the latest trade for each coin
import os as _os
_os.makedirs('/data/trading28/charts', exist_ok=True)

charts = []
for sym, (c8, h8, l8, o8, idx, sa, sb, t_arr, k_arr, trades) in sym_trades.items():
    best = max(trades, key=lambda t: abs(t['pnl']))
    fname = f'/data/trading28/charts/{sym}_trade.png'
    plot_trade(sym, c8, h8, l8, o8, idx, sa, sb, t_arr, k_arr, best, fname)
    charts.append(fname)
    print(f"✅ {sym}: {best['pnl']:+.1f}% | {best['entry_time'].strftime('%m/%d %H:%M')} → {best['exit_time'].strftime('%m/%d %H:%M')} | {fname}")

print(f"\n📊 {len(charts)} charts saved to /data/trading28/charts/")
