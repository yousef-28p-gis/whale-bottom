#!/usr/bin/env python3
"""Cloud Hunter — Last month: Jul 3 - Aug 3, 2026"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load(sym):
    p = os.path.join('/data/trading28/data/whale_15m_1y', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    ts = j.get('ts', [])
    if not ts: return None
    # Filter to last month only
    c = np.array(j['c'], float); h = np.array(j['h'], float)
    l = np.array(j['l'], float); o = np.array(j['o'], float)
    mask = np.array(ts) >= CUTOFF
    if mask.sum() < 200: return None  # need enough candles
    return (c[mask], h[mask], l[mask], o[mask], [t for i,t in enumerate(ts) if mask[i]])

def resample_8h(c, h, l, o, ts):
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o,'h':h,'l':l,'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except: return None

def ichimoku_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
    n = len(c)
    if n < senkou + 30: return []
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
    
    trades = []
    pos = 0; ep = 0; cool = 0; entry_idx = None
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i])
        above = c[i] > cloud_top
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        
        if pos:
            if h[i] >= ep * (1 + tp/100):
                trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
            elif l[i] <= ep * (1 - sl/100):
                pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0:
            if above and golden: pos = 1; ep = c[i]; entry_idx = idx[i]
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades

def run_2positions(coin_trades):
    eq = 1000; eq_curve = [1000]; open_positions = {}
    timeline = []
    for sym, trades in coin_trades.items():
        for entry_t, exit_t, pnl in trades:
            timeline.append((entry_t, 'entry', sym, pnl))
            timeline.append((exit_t, 'exit', sym, pnl))
    timeline.sort()
    executed = 0; wins = 0
    for t, etype, sym, pnl in timeline:
        if etype == 'entry':
            if len(open_positions) < 2:
                open_positions[sym] = eq / 2
        elif etype == 'exit':
            if sym in open_positions:
                alloc = open_positions.pop(sym)
                new_val = alloc * (1 + pnl/100)
                eq += (new_val - alloc)
                eq_curve.append(eq)
                executed += 1
                if pnl > 0: wins += 1
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    wr = wins/executed*100 if executed else 0
    return {'pnl': eq-1000, 'dd': dd, 'trades': executed, 'wr': wr, 'eq': eq}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Find clean coins
coin_trades_all = {}
for sym in tradeable:
    data = load(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades = ichimoku_trades(c8, h8, l8, o8, idx)
    if len(trades) >= 1:
        coin_trades_all[sym] = trades

# Also test: top 60 by total historical PnL
coin_pnls_hist = {}
for sym in coin_trades_all:
    coin_pnls_hist[sym] = sum(p for _, _, p in coin_trades_all[sym])

ranked = sorted(coin_pnls_hist.items(), key=lambda x: x[1], reverse=True)
top60 = dict(ranked[:60])
top30 = dict(ranked[:30])

print(f"☁️ صياد السحابة — آخر شهر (يوليو 2026)\n")

total_avail = sum(len(v) for v in coin_trades_all.values())
print(f"📊 كل العملات المتاحة:")
m = run_2positions(coin_trades_all)
print(f"   {len(coin_trades_all)} عملة | {total_avail} إشارة | منفذ={m['trades']} | {m['wr']:.1f}% WR | ${m['pnl']:+,.0f} | سحب={m['dd']:.1f}% | نهائي=${m['eq']:,.0f}")

top60_trades = {s: coin_trades_all[s] for s in top60 if s in coin_trades_all}
m60 = run_2positions(top60_trades)
avail60 = sum(len(v) for v in top60_trades.values())
print(f"\n📊 أفضل 60 عملة:")
print(f"   {len(top60_trades)} عملة | {avail60} إشارة | منفذ={m60['trades']} | {m60['wr']:.1f}% WR | ${m60['pnl']:+,.0f} | سحب={m60['dd']:.1f}% | نهائي=${m60['eq']:,.0f}")

top30_trades = {s: coin_trades_all[s] for s in top30 if s in coin_trades_all}
m30 = run_2positions(top30_trades)
avail30 = sum(len(v) for v in top30_trades.values())
print(f"\n📊 أفضل 30 عملة:")
print(f"   {len(top30_trades)} عملة | {avail30} إشارة | منفذ={m30['trades']} | {m30['wr']:.1f}% WR | ${m30['pnl']:+,.0f} | سحب={m30['dd']:.1f}% | نهائي=${m30['eq']:,.0f}")

# Show individual results for top 60
print(f"\n🏆 أفضل 10 عملات في الشهر:")
for sym in sorted(top60_trades.keys(), key=lambda s: sum(p for _,_,p in top60_trades[s]), reverse=True)[:10]:
    trades = top60_trades[sym]
    pnl = sum(p for _,_,p in trades)
    wins = sum(1 for _,_,p in trades if p > 0)
    wr = wins/len(trades)*100 if trades else 0
    latest_date = trades[-1][1].strftime('%m/%d') if trades else '—'
    print(f"  {sym:>8s} | {len(trades):2d} صفقة | WR={wr:.0f}% | ${pnl:+,.0f} | آخر={latest_date}")

print(f"\n👎 أسوأ 5:")
for sym in sorted(top60_trades.keys(), key=lambda s: sum(p for _,_,p in top60_trades[s]))[:5]:
    trades = top60_trades[sym]
    pnl = sum(p for _,_,p in trades)
    print(f"  {sym:>8s} | {len(trades):2d} صفقة | ${pnl:+,.0f}")
