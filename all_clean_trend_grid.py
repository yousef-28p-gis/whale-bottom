#!/usr/bin/env python3
"""All clean coins: 2 positions $500 — comprehensive trend filter grid"""
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

def ichimoku_trades(c, h, l, o, idx, ft=None):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
    n = len(c)
    if n < senkou + 30: return []
    
    # Precompute all MAs we might need
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    ema100 = pd.Series(c).ewm(span=100, adjust=False).mean().values
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
    sma50 = pd.Series(c).rolling(50).mean().values
    sma100 = pd.Series(c).rolling(100).mean().values
    sma200 = pd.Series(c).rolling(200).mean().values
    
    # ADX-like: directional movement over 14 periods
    tr = np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1]))
    tr = np.maximum(tr, np.abs(l[1:] - c[:-1]))
    atr = pd.Series(np.insert(tr, 0, np.nan)).rolling(14).mean().values
    
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
        
        # Trend filters
        long_ok = True; short_ok = True
        if ft == 'ema50':
            long_ok = c[i] > ema50[i]; short_ok = c[i] < ema50[i]
        elif ft == 'ema100':
            long_ok = c[i] > ema100[i]; short_ok = c[i] < ema100[i]
        elif ft == 'ema200':
            long_ok = c[i] > ema200[i]; short_ok = c[i] < ema200[i]
        elif ft == 'sma50_100':
            if not (np.isnan(sma50[i]) or np.isnan(sma100[i])):
                long_ok = sma50[i] > sma100[i]; short_ok = sma50[i] < sma100[i]
        elif ft == 'sma50_200':
            if not (np.isnan(sma50[i]) or np.isnan(sma200[i])):
                long_ok = sma50[i] > sma200[i]; short_ok = sma50[i] < sma200[i]
        elif ft == 'ema50_100':
            long_ok = ema50[i] > ema100[i]; short_ok = ema50[i] < ema100[i]
        elif ft == 'ema50_200':
            long_ok = ema50[i] > ema200[i]; short_ok = ema50[i] < ema200[i]
        elif ft == 'above_cloud_only':
            short_ok = False  # only long
        
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

# Load
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Identify bad coins
coin_pp = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            coin_pp[sym] = coin_pp.get(sym, 0) + sum(p for _, _, p in trades)

exclude = set()
for sym, total_pnl in coin_pp.items():
    if total_pnl < 0: exclude.add(sym)

# Calculate per-period negativity
coin_period_pnl = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            pnl = sum(p for _, _, p in trades)
            if sym not in coin_period_pnl: coin_period_pnl[sym] = {}
            coin_period_pnl[sym][pdir] = pnl

# Exclude if negative in 2+ periods
for sym, pp in coin_period_pnl.items():
    neg = sum(1 for p in pp.values() if p < 0)
    if neg >= 2: exclude.add(sym)

clean = set(tradeable) - exclude
print(f"🧹 عملات نظيفة: {len(clean)} | مستبعدة: {len(exclude)}\n")

# Filters to test
filters = {
    None: 'بدون فلتر',
    'ema50': 'EMA50',
    'ema100': 'EMA100',
    'ema200': 'EMA200',
    'ema50_100': 'EMA50>100',
    'ema50_200': 'EMA50>200',
    'sma50_100': 'SMA50>100',
    'sma50_200': 'SMA50>200',
    'above_cloud_only': 'فوق السحابة فقط',
}

# Test on all 3 periods
print(f"{'فلتر':>16s} | {'2023':>15s} | {'PREV':>15s} | {'CUR':>15s} | {'المجموع':>12s}")
print(f"{'─'*16}┼{'─'*17}┼{'─'*17}┼{'─'*17}┼{'─'*14}")

for ft, fname in filters.items():
    row = f"{fname:>16s} |"
    total = 0
    for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
        coin_trades = {}
        for sym in clean:
            data = load(sym, pdir)
            if data is None: continue
            resampled = resample_8h(*data)
            if resampled is None: continue
            c8, h8, l8, o8, idx = resampled
            trades = ichimoku_trades(c8, h8, l8, o8, idx, ft)
            if len(trades) >= 3:
                coin_trades[sym] = trades
        
        if not coin_trades:
            row += f" {'—':>15s} |"
            continue
        
        m = run_2positions(coin_trades)
        total += m['pnl']
        row += f" ${m['pnl']:+6.0f} DD={m['dd']:4.1f}% |"
    
    row += f" ${total:+10,.0f}"
    print(row)
