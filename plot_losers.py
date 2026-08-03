#!/usr/bin/env python3
"""Find and plot losing Cloud Hunter trades from July 2026"""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2
CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load_last_month(sym):
    p = os.path.join('/data/trading28/data/whale_15m_1y', sym + '.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    ts = j.get('ts', [])
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

def get_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18
    n = len(c)
    if n < senkou + 30: return None
    h_t = pd.Series(h).rolling(tenkan).max().values
    l_t = pd.Series(l).rolling(tenkan).min().values
    t_arr = (h_t + l_t) / 2
    h_k = pd.Series(h).rolling(kijun).max().values
    l_k = pd.Series(l).rolling(kijun).min().values
    k_arr = (h_k + l_k) / 2
    h_s = pd.Series(h).rolling(senkou).max().values
    l_s = pd.Series(l).rolling(senkou).min().values
    sb_raw = (h_s + l_s) / 2
    sa_raw = (t_arr + k_arr) / 2
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n:
            sa[i+shift] = sa_raw[i]
            sb[i+shift] = sb_raw[i]
    cloud_top = np.maximum(sa, sb)
    trades = []
    pos = 0; ep = 0; cool = 0; entry_bar = 0
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]): continue
        above = c[i] > cloud_top[i]
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        if pos:
            if h[i] >= ep * 1.05:
                trades.append({'e': entry_bar, 'x': i, 'ep': ep, 'xp': c[i], 'pnl': 5-COMM*100, 'r': 'TP'})
                pos = 0; cool = COOLDOWN
            elif l[i] <= ep * 0.975:
                pnl = -2.5 - COMM*100  # fixed -2.5% + commission
                trades.append({'e': entry_bar, 'x': i, 'ep': ep, 'xp': c[i], 'pnl': pnl, 'r': 'SL'})
                pos = 0; cool = COOLDOWN
        if not pos and cool == 0 and above and golden:
            pos = 1; ep = c[i]; entry_bar = i
        if not pos and cool > 0: cool -= 1
    if pos:
        trades.append({'e': entry_bar, 'x': n-1, 'ep': ep, 'xp': c[-1], 'pnl': (c[-1]/ep-1)*100-COMM*100, 'r': 'OPEN'})
    return trades, sa, sb, t_arr, k_arr

def plot_one(sym, c, h, l, o, idx, sa, sb, t_arr, k_arr, t, fname):
    s = max(0, t['e'] - 25)
    e = min(len(c), t['x'] + 12)
    cs = c[s:e]; hs = h[s:e]; ls = l[s:e]; os = o[s:e]
    ids = idx[s:e]; xx = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor('white')
    colors = ['#26a69a' if cs[i] >= os[i] else '#ef5350' for i in range(len(cs))]
    bw = 0.4
    for i in range(len(cs)):
        ax.plot([i,i], [ls[i], hs[i]], color=colors[i], linewidth=1)
        ax.add_patch(plt.Rectangle((i-bw/2, min(os[i],cs[i])), bw, abs(cs[i]-os[i]),
                                    facecolor=colors[i], edgecolor=colors[i], alpha=0.9))
    sa_s = sa[s:e]; sb_s = sb[s:e]
    ax.fill_between(xx, sa_s, sb_s, where=~np.isnan(sa_s), alpha=0.15, color='#4CAF50')
    ax.plot(xx, t_arr[s:e], color='#2196F3', lw=1, alpha=0.6)
    ax.plot(xx, k_arr[s:e], color='#FF9800', lw=1, alpha=0.6)
    ex = t['e'] - s
    ax.scatter(ex, t['ep'], color='lime', s=200, zorder=5, marker='^', edgecolors='darkgreen', linewidths=2)
    ax.scatter(t['x'] - s, t['xp'], color='#f44336', s=200, zorder=5, marker='v', edgecolors='black', linewidths=2)
    ax.axhline(t['ep']*0.975, xmin=ex/len(cs), xmax=(t['x']-s)/len(cs), color='red', ls='--', alpha=0.3)
    ax.axhline(t['ep']*1.05, xmin=ex/len(cs), xmax=(t['x']-s)/len(cs), color='green', ls='--', alpha=0.3)
    ticks = range(0, len(cs), max(1, len(cs)//8))
    ax.set_xticks(ticks)
    ax.set_xticklabels([ids[i].strftime('%m/%d %Hh') for i in ticks], rotation=45, fontsize=7)
    ax.set_title(sym + ' | SL: ' + str(round(t['pnl'],1)) + '%', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)

os.makedirs('/data/trading28/charts', exist_ok=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

losers = []
for sym in tradeable:
    data = load_last_month(sym)
    if data is None: continue
    rp = resample_8h(*data)
    if rp is None: continue
    result = get_trades(*rp)
    if result is None: continue
    trades, sa, sb, t_arr, k_arr = result
    for t in trades:
        if t['pnl'] < 0:
            losers.append((sym,) + rp + (sa, sb, t_arr, k_arr, t))

losers.sort(key=lambda x: x[-1]['pnl'])
print('Found', len(losers), 'losing trades')
for i, items in enumerate(losers[:6]):
    sym = items[0]
    c8, h8, l8, o8, idx = items[1], items[2], items[3], items[4], items[5]
    sa, sb, t_arr, k_arr = items[6], items[7], items[8], items[9]
    t = items[10]
    fname = '/data/trading28/charts/LOSE_' + str(i+1) + '_' + sym + '.png'
    plot_one(sym, c8, h8, l8, o8, idx, sa, sb, t_arr, k_arr, t, fname)
    print(i+1, sym, round(t['pnl'],1), '%', fname)
