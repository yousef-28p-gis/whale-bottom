#!/usr/bin/env python3
"""Find worst coins: highest DD + lowest/negative PnL across 3 periods"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; MAX_SLIPPAGE = 1.5

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
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values
    except: return None

def run_coin(c, h, l, o):
    tenkan, kijun, senkou = 3, 9, 18
    tp, sl, cooldown = 5, 2.5, 2
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
    sb_raw = (h_s + l_s) / 2; sa_raw = (t_arr + k_arr) / 2
    
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n: sa[i+shift] = sa_raw[i]; sb[i+shift] = sb_raw[i]
    
    pnls = []; eq = 1000; cv = [1000]
    pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    pnl = tp - COMM*100; pnls.append(pnl); eq *= (1+pnl/100); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep-1)*100-COMM*100, -sl*MAX_SLIPPAGE-COMM*100)
                    pnls.append(pnl); eq *= (1+pnl/100); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    pnl = tp - COMM*100; pnls.append(pnl); eq *= (1+pnl/100); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1-c[i]/ep)*100-COMM*100, -sl*MAX_SLIPPAGE-COMM*100)
                    pnls.append(pnl); eq *= (1+pnl/100); pos = 0; cool = cooldown
        
        if not pos and cool == 0:
            if above and golden: pos = 1; ep = c[i]; side = 1
            elif below_cloud and death: pos = 1; ep = c[i]; side = -1
        if not pos and cool > 0: cool -= 1
        cv.append(eq)
    
    if pos:
        pnl = (c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        pnls.append(pnl); eq *= (1+pnl/100)
    
    if len(pnls) < 3: return None
    s = pd.Series(cv)
    dd = ((s - s.expanding().max()) / s.expanding().max() * 100).min()
    w = sum(1 for p in pnls if p > 0)
    return {'trades': len(pnls), 'wr': w/len(pnls)*100, 'pnl': eq-1000, 'dd': dd, 'pnls': pnls}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Collect all coins across all periods
coin_stats = {}  # sym -> {period: stats}

PERIOD_MAP = {'2023': '2023', 'PREV': 'prev', 'CUR': 'cur'}

for period_name, dir_suffix in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    pkey = period_name
    for sym in tradeable:
        data = load(sym, dir_suffix)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8 = resampled
        r = run_coin(c8, h8, l8, o8)
        if r is None: continue
        if sym not in coin_stats: coin_stats[sym] = {}
        coin_stats[sym][pkey] = r

# Find worst: highest DD + negative PnL in most periods
worst_list = []
for sym, periods in coin_stats.items():
    max_dd = min(r['dd'] for r in periods.values())
    pnls = {p: r['pnl'] for p, r in periods.items()}
    avg_pnl = np.mean(list(pnls.values()))
    num_periods = len(periods)
    neg_periods = sum(1 for p in pnls.values() if p < 0)
    
    if avg_pnl < 0 or max_dd < -30:
        worst_list.append({
            'sym': sym,
            'periods': num_periods,
            'neg_periods': neg_periods,
            'avg_pnl': avg_pnl,
            'max_dd': max_dd,
            'pnls': pnls,
        })

# Sort by DD (worst first) then by avg PnL
worst_list.sort(key=lambda x: (x['max_dd'], x['avg_pnl']))

print("🔴 العملات الأسوأ — سحب عالي + ربح سلبي\n")
print(f"{'عملة':>8s} | {'فترات':>4s} | {'خاسرة':>5s} | {'سحب':>7s} | {'ربح 2023':>9s} | {'ربح PREV':>9s} | {'ربح CUR':>9s}")
print("─" * 75)

for w in worst_list[:25]:
    p = w['pnls']
    p23 = f"{p.get('2023', 0):+,.0f}"
    pp = f"{p.get('PREV', 0):+,.0f}"
    pc = f"{p.get('CUR', 0):+,.0f}"
    print(f"{w['sym']:>8s} | {w['periods']:4d} | {w['neg_periods']:5d} | {w['max_dd']:6.1f}% | {p23:>9s} | {pp:>9s} | {pc:>9s}")

print(f"\n📊 إجمالي العملات في القائمة: {len(worst_list)}")
