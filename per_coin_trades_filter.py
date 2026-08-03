#!/usr/bin/env python3
"""Per-coin trade counts with SMA50>100 filter"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

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

def ichimoku_trades(c, h, l, o, idx, use_filter=True):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
    n = len(c)
    if n < senkou + 30: return [], None
    
    sma50 = pd.Series(c).rolling(50).mean().values
    sma100 = pd.Series(c).rolling(100).mean().values
    
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
        
        long_ok = True; short_ok = True
        if use_filter:
            if not (np.isnan(sma50[i]) or np.isnan(sma100[i])):
                long_ok = sma50[i] > sma100[i]
                short_ok = sma50[i] < sma100[i]
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
            else:
                if l[i] <= ep * (1 - tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0:
            if above and golden and long_ok:
                pos = 1; ep = c[i]; side = 1; entry_idx = idx[i]
            elif below_cloud and death and short_ok:
                pos = 1; ep = c[i]; side = -1; entry_idx = idx[i]
        
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    
    # Per-coin metrics
    pnls = [p for _, _, p in trades]
    if len(pnls) < 3: return [], None
    w = sum(1 for p in pnls if p > 0)
    eq = 1000
    for p in pnls: eq *= (1 + p/100)
    dd_cv = [1000]
    eq_tmp = 1000
    for p in pnls:
        eq_tmp *= (1 + p/100)
        dd_cv.append(eq_tmp)
    s = pd.Series(dd_cv); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    
    info = {
        'trades': len(pnls),
        'wr': w/len(pnls)*100,
        'pnl': eq - 1000,
        'dd': dd,
    }
    return trades, info

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Find bad coins
coin_pp = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades, _ = ichimoku_trades(c8, h8, l8, o8, idx, False)
        if trades:
            coin_pp[sym] = coin_pp.get(sym, 0) + sum(p for _, _, p in trades)

# Per-period negativity
cpp = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades, _ = ichimoku_trades(c8, h8, l8, o8, idx, False)
        if trades:
            pnl = sum(p for _, _, p in trades)
            if sym not in cpp: cpp[sym] = {}
            cpp[sym][pdir] = pnl

exclude = set()
for sym, pp in cpp.items():
    neg = sum(1 for p in pp.values() if p < 0)
    if neg >= 2: exclude.add(sym)

clean = sorted(set(tradeable) - exclude)
print(f"عملات نظيفة: {len(clean)}\n")

# Per-coin analysis with SMA50>100 filter
print(f"{'عملة':>8s} | {'2023 صفقات':>10s} | {'PREV صفقات':>10s} | {'CUR صفقات':>10s} | {'إجمالي صفقات':>10s} | {'إجمالي ربح$':>10s} | {'سحب':>6s}")
print("─" * 85)

grand_trades = 0
results = []

for sym in clean:
    row = f"{sym:>8s} |"
    total_trades = 0; total_pnl = 0
    worst_dd = 0
    
    for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
        data = load(sym, pdir)
        if data is None:
            row += f" {'—':>10s} |"
            continue
        resampled = resample_8h(*data)
        if resampled is None:
            row += f" {'—':>10s} |"
            continue
        c8, h8, l8, o8, idx = resampled
        trades, info = ichimoku_trades(c8, h8, l8, o8, idx, True)
        if info is None:
            row += f" {'0':>10s} |"
            continue
        
        n = info['trades']
        total_trades += n
        total_pnl += info['pnl']
        worst_dd = min(worst_dd, info['dd'])
        row += f" {n:>10d} |"
    
    row += f" {total_trades:>10d} | {total_pnl:+10,.0f}$ | {worst_dd:5.1f}%"
    grand_trades += total_trades
    results.append((total_pnl, total_trades, worst_dd, sym, row))

# Sort by PnL
results.sort(key=lambda x: x[0], reverse=True)

print(f"\n🏆 الأعلى ربحاً:")
for pnl, trades, dd, sym, row in results[:15]:
    print(row)

print(f"\n👎 الأقل ربحاً:")
for pnl, trades, dd, sym, row in results[-10:]:
    print(row)

print(f"\n📊 إجمالي الصفقات: {grand_trades}")
print(f"📊 متوسط الصفقات لكل عملة: {grand_trades/len(results):.0f}")
