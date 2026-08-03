#!/usr/bin/env python3
"""Compare: 3-year top 60 vs last-month top 60"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2
CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load(sym, period):
    p = os.path.join(f'/data/trading28/data/whale_15m_{period}', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

def load_last_month(sym):
    data = load(sym, '1y')
    if data is None: return None
    c, h, l, o, ts = data
    mask = np.array(ts) >= CUTOFF
    if mask.sum() < 200: return None
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
        above = c[i] > max(sa[i], sb[i])
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
            if len(open_positions) < 2: open_positions[sym] = eq / 2
        elif etype == 'exit':
            if sym in open_positions:
                alloc = open_positions.pop(sym)
                eq += alloc * (1 + pnl/100) - alloc
                eq_curve.append(eq); executed += 1
                if pnl > 0: wins += 1
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    return {'pnl': eq-1000, 'dd': dd, 'trades': executed, 'wr': wins/executed*100 if executed else 0, 'eq': eq}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# ── Rank by 3-year PnL (long only) ──
coin_3y_pnl = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            coin_3y_pnl[sym] = coin_3y_pnl.get(sym, 0) + sum(p for _,_,p in trades)

ranked_3y = sorted(coin_3y_pnl.items(), key=lambda x: x[1], reverse=True)
top60_3y = set(c for c, _ in ranked_3y[:60])
top30_3y = set(c for c, _ in ranked_3y[:30])

# ── Rank by last month PnL ──
coin_lm_pnl = {}
for sym in tradeable:
    data = load_last_month(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades = ichimoku_trades(c8, h8, l8, o8, idx)
    if len(trades) >= 1:
        coin_lm_pnl[sym] = sum(p for _,_,p in trades)

ranked_lm = sorted(coin_lm_pnl.items(), key=lambda x: x[1], reverse=True)
top60_lm = set(c for c, _ in ranked_lm[:60])
top30_lm = set(c for c, _ in ranked_lm[:30])

# ── Compare ──
overlap_60 = top60_3y & top60_lm
overlap_30 = top30_3y & top30_lm

print(f"🔍 مقارنة: ترتيب 3 سنوات vs ترتيب آخر شهر\n")
print(f"أفضل 60 — 3 سنوات: {len(top60_3y)}")
print(f"أفضل 60 — آخر شهر: {len(top60_lm)}")
print(f"🔄 التداخل: {len(overlap_60)} عملة مشتركة فقط! ({len(overlap_60)/60*100:.0f}%)")
print(f"   المشتركة: {', '.join(sorted(overlap_60)[:15])}...")

print(f"\nأفضل 30 — 3 سنوات: {len(top30_3y)}")
print(f"أفضل 30 — آخر شهر: {len(top30_lm)}")
print(f"🔄 التداخل: {len(overlap_30)} عملة مشتركة فقط! ({len(overlap_30)/30*100:.0f}%)")
print(f"   المشتركة: {', '.join(sorted(overlap_30))}")

# ── Now run last month with 3-year top 60 ──
print(f"\n{'='*60}")
print(f"📊 اختبار آخر شهر باستخدام أفضل 60 عملة من 3 سنوات:")
coin_trades_3y60 = {}
for sym in top60_3y:
    data = load_last_month(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades = ichimoku_trades(c8, h8, l8, o8, idx)
    if len(trades) >= 1:
        coin_trades_3y60[sym] = trades

total_avail = sum(len(v) for v in coin_trades_3y60.values())
m = run_2positions(coin_trades_3y60)
print(f"   {len(coin_trades_3y60)} عملة | {total_avail} إشارة | منفذ={m['trades']} | {m['wr']:.1f}% WR")
print(f"   ${m['pnl']:+,.0f} | سحب={m['dd']:.1f}% | نهائي=${m['eq']:,.0f}")

# Last month's OWN top 60 for comparison
coin_trades_lm60 = {}
for sym in top60_lm:
    data = load_last_month(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades = ichimoku_trades(c8, h8, l8, o8, idx)
    if len(trades) >= 1:
        coin_trades_lm60[sym] = trades

avail_lm = sum(len(v) for v in coin_trades_lm60.values())
m2 = run_2positions(coin_trades_lm60)
print(f"\n📊 (للمقارنة) أفضل 60 حسب الشهر نفسه (cherry-picked):")
print(f"   {len(coin_trades_lm60)} عملة | {avail_lm} إشارة | منفذ={m2['trades']} | {m2['wr']:.1f}% WR")
print(f"   ${m2['pnl']:+,.0f} | سحب={m2['dd']:.1f}% | نهائي=${m2['eq']:,.0f}")
