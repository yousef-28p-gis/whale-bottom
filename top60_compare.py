#!/usr/bin/env python3
"""Top 60 coins: equal $11 vs 2 positions $500"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

def load(sym, period):
    p = os.path.join(f'/data/trading28/data/whale_15m_{period}', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

def resample_8h(c, h, l, o, ts):
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o,'h':h,'l':l,'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except: return None

def ichimoku_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl, cooldown = 5, 2.5, 2
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
    pos = 0; ep = 0; cool = 0; side = 0; entry_idx = None
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = cooldown
        if not pos and cool == 0:
            if above and golden: pos = 1; ep = c[i]; side = 1; entry_idx = idx[i]
            elif below_cloud and death: pos = 1; ep = c[i]; side = -1; entry_idx = idx[i]
        if not pos and cool > 0: cool -= 1
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Step 1: Rank all coins by total PnL across all periods
print("🔍 ترتيب العملات حسب إجمالي الربح...")
coin_pnls = {}
for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            if sym not in coin_pnls: coin_pnls[sym] = 0
            coin_pnls[sym] += sum(p for _, _, p in trades)

# Sort and pick top 60
ranked = sorted(coin_pnls.items(), key=lambda x: x[1], reverse=True)
top60 = set(c for c, _ in ranked[:60])
print(f"🏆 أقوى 60 عملة: {', '.join(sorted(top60)[:15])}...")

# Step 2: Run both methods
for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    print(f"\n{'='*50}")
    print(f"📅 {pname}")
    
    coin_trades = {}
    for sym in top60:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            coin_trades[sym] = trades
    
    N = len(coin_trades)
    
    # Method 1: $11 per coin
    per_coin = 11  # fixed $11
    coin_eq = {s: per_coin for s in coin_trades}
    events = []
    for sym, trades in coin_trades.items():
        for entry, exit_t, pnl in trades:
            events.append((exit_t, sym, pnl))
    events.sort()
    pf_eq = sum(coin_eq.values())
    eq_curve = [pf_eq]
    for _, sym, pnl in events:
        old = coin_eq[sym]; coin_eq[sym] *= (1 + pnl/100)
        pf_eq += (coin_eq[sym] - old); eq_curve.append(pf_eq)
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd1 = ((s - peak) / peak * 100).min()
    total1 = len(events); wins1 = sum(1 for _, _, p in events if p > 0)
    wr1 = wins1/total1*100 if total1 else 0
    
    total_capital = N * 11
    print(f"\n📊 توزيع متساوي: {N} عملة × $11 = ${total_capital}")
    print(f"   {total1} صفقة | WR={wr1:.1f}% | ${pf_eq-total_capital:+,.0f} | سحب={dd1:.1f}% | نهائي=${pf_eq:,.0f}")
    
    # Method 2: Max 2 positions × $500 each
    eq2 = 1000; eq_curve2 = [1000]
    open_positions = {}  # sym -> equity_allocated
    # Build timeline
    timeline = []
    for sym, trades in coin_trades.items():
        for entry_t, exit_t, pnl in trades:
            timeline.append((entry_t, 'entry', sym, pnl))
            timeline.append((exit_t, 'exit', sym, pnl))
    timeline.sort()
    
    executed = 0; wins2 = 0
    for t, etype, sym, pnl in timeline:
        if etype == 'entry':
            if len(open_positions) < 2:
                alloc = eq2 / 2
                open_positions[sym] = alloc
            # else: skip — no free slots
        elif etype == 'exit':
            if sym in open_positions:
                alloc = open_positions.pop(sym)
                new_val = alloc * (1 + pnl/100)
                eq2 += (new_val - alloc)
                eq_curve2.append(eq2)
                executed += 1
                if pnl > 0: wins2 += 1
    
    s2 = pd.Series(eq_curve2)
    peak2 = s2.expanding().max()
    dd2 = ((s2 - peak2) / peak2 * 100).min()
    wr2 = wins2/executed*100 if executed else 0
    
    print(f"\n📊 صفقتين (${500} لكل مركز):")
    print(f"   {executed} صفقة منفذة | WR={wr2:.1f}% | ${eq2-1000:+,.0f} | سحب={dd2:.1f}% | نهائي=${eq2:,.0f}")
