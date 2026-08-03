#!/usr/bin/env python3
"""Cloud Hunter — test different exit methods, last month — FIXED"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2
CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)

def load_last_month(sym):
    p = os.path.join('/data/trading28/data/whale_15m_1y', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    ts = j.get('ts', []); c = np.array(j['c'], float)
    h = np.array(j['h'], float); l = np.array(j['l'], float)
    o = np.array(j['o'], float)
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
    return entries, sa, sb

def run_trades(entries, c, h, l, o, idx, exit_mode, ema20, sa, sb):
    n = len(c); trades = []; pos = 0; ep = 0; cool = 0; entry_idx = None
    
    for i in range(n):
        exit_signal = False; pnl = 0
        
        if pos:
            highest_since_entry = max(h[entry_bar:i+1]) if entry_bar <= i else ep
            
            if exit_mode == 'fixed':
                if h[i] >= ep * 1.05: pnl = 5 - COMM*100; exit_signal = True
                elif l[i] <= ep * 0.975: pnl = max((c[i]/ep-1)*100 - COMM*100, -2.5*MAX_SLIPPAGE-COMM*100); exit_signal = True
            
            elif exit_mode == 'trailing':
                if l[i] <= highest_since_entry * 0.975:
                    pnl = max((highest_since_entry/ep - 1)*100 - COMM*100, -2.5*MAX_SLIPPAGE-COMM*100)
                    exit_signal = True
            
            elif exit_mode == 'ema_exit':
                if c[i] < ema20[i] and not np.isnan(ema20[i]):
                    pnl = (c[i]/ep - 1)*100 - COMM*100; exit_signal = True
            
            elif exit_mode == 'cloud_exit':
                if not np.isnan(sa[i]) and not np.isnan(sb[i]):
                    if c[i] < max(sa[i], sb[i]):
                        pnl = (c[i]/ep - 1)*100 - COMM*100; exit_signal = True
            
            elif exit_mode == 'hybrid':
                # TP at 5%, else trailing 2.5% stop
                if h[i] >= ep * 1.05: pnl = 5 - COMM*100; exit_signal = True
                elif l[i] <= highest_since_entry * 0.975:
                    pnl = max((highest_since_entry/ep - 1)*100 - COMM*100, -2.5*MAX_SLIPPAGE-COMM*100)
                    exit_signal = True
            
            elif exit_mode == 'breakeven':
                # TP 5%, SL 2.5%, move SL to breakeven after 3% gain
                if h[i] >= ep * 1.05: pnl = 5 - COMM*100; exit_signal = True
                elif highest_since_entry >= ep * 1.03:
                    if l[i] <= ep * 1.001:  # breakeven
                        pnl = max((ep/ep-1)*100 - COMM*100, -0.5); exit_signal = True
                elif l[i] <= ep * 0.975:
                    pnl = -2.5*MAX_SLIPPAGE - COMM*100; exit_signal = True
            
            if exit_signal:
                trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0 and entries[i]:
            pos = 1; ep = c[i]; entry_idx = idx[i]; entry_bar = i
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

# Load
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Get 3-year top 60
coin_3y = {}
for pdir in ['2023','prev','1y']:
    p = f'/data/trading28/data/whale_15m_{pdir}'
    for sym in tradeable:
        fp = os.path.join(p, f'{sym}.json')
        if not os.path.exists(fp): continue
        with open(fp) as f: j = json.load(f)
        c = np.array(j['c'], float); h = np.array(j['h'], float)
        l = np.array(j['l'], float); o = np.array(j['o'], float); ts = j.get('ts', [])
        rp = resample_8h(c, h, l, o, ts)
        if rp is None: continue
        c8, h8, l8, o8, idx = rp
        entries, sa, sb = get_entries(c8, h8, l8, o8)
        ema20 = pd.Series(c8).ewm(span=20).mean().values
        trades = run_trades(entries, c8, h8, l8, o8, idx, 'fixed', ema20, sa, sb)
        if len(trades) >= 3:
            coin_3y[sym] = coin_3y.get(sym, 0) + sum(p for _,_,p in trades)

top60 = set(c for c, _ in sorted(coin_3y.items(), key=lambda x: x[1], reverse=True)[:60])

# Load last month
coin_data = {}
for sym in tradeable:
    data = load_last_month(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    coin_data[sym] = resampled

exit_methods = [
    ('TP5/SL2.5 ثابت', 'fixed'),
    ('تريلينغ 2.5%', 'trailing'),
    ('خروج تحت EMA20', 'ema_exit'),
    ('خروج تحت السحابة', 'cloud_exit'),
    ('TP5 + تريلينغ SL', 'hybrid'),
    ('TP5 + تعادل بعد 3%', 'breakeven'),
]

print(f"☁️ صياد السحابة — طرق خروج مختلفة | أفضل 60 | يوليو 2026\n")
print(f"{'طريقة الخروج':>24s} | {'صفقات':>4s} | {'WR':>5s} | {'ربح$':>7s} | {'سحب':>6s}")
print("─" * 62)

for label, mode in exit_methods:
    coin_trades = {}
    for sym in top60 & set(coin_data.keys()):
        c8, h8, l8, o8, idx = coin_data[sym]
        entries, sa, sb = get_entries(c8, h8, l8, o8)
        ema20 = pd.Series(c8).ewm(span=20).mean().values
        trades = run_trades(entries, c8, h8, l8, o8, idx, mode, ema20, sa, sb)
        if len(trades) >= 1:
            coin_trades[sym] = trades
    
    if not coin_trades: continue
    m = run_2positions(coin_trades)
    print(f"{label:>24s} | {m['trades']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+6,.0f} | {m['dd']:5.1f}%")
