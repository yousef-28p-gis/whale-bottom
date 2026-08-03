#!/usr/bin/env python3
"""Test multiple strategies — last month, long only, 2 positions $500"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone

COMM = 0.002; MAX_SLIPPAGE = 1.5

CUTOFF = int(datetime(2026, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)
TP, SL, COOLDOWN = 5, 2.5, 2

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
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o,'h':h,'l':l,'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except: return None

def run_trades(entries, exits, c, h, l, o, idx):
    """Generic: entries=bool array, exits based on TP/SL tracking"""
    n = len(c)
    trades = []; pos = 0; ep = 0; cool = 0; entry_idx = None
    
    for i in range(n):
        if pos:
            if h[i] >= ep * (1 + TP/100):
                trades.append((entry_idx, idx[i], TP - COMM*100)); pos = 0; cool = COOLDOWN
            elif l[i] <= ep * (1 - SL/100):
                pnl = max((c[i]/ep - 1)*100 - COMM*100, -SL*MAX_SLIPPAGE - COMM*100)
                trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0 and entries[i]:
            pos = 1; ep = c[i]; entry_idx = idx[i]
        
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
                eq_curve.append(eq)
                executed += 1
                if pnl > 0: wins += 1
    
    s = pd.Series(eq_curve)
    peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    
    return {'pnl': eq-1000, 'dd': dd, 'trades': executed, 'wr': wins/executed*100 if executed else 0, 'eq': eq}

# ── Strategies ──
def ichimoku_entries(c, h, l, o):
    tenkan, kijun, senkou = 3, 9, 18; n = len(c)
    if n < senkou + 30: return np.zeros(n, bool)
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
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        above = c[i] > max(sa[i], sb[i])
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        if above and golden: entries[i] = True
    return entries

def breakout_entries(c, h, l, o, lookback=20):
    n = len(c)
    entries = np.zeros(n, bool)
    if n < lookback + 2: return entries
    for i in range(lookback, n):
        highest = max(h[i-lookback:i])
        if c[i] > highest and c[i-1] <= h[i-lookback:i-1].max():
            entries[i] = True
    return entries

def ema_cross_entries(c, h, l, o, fast=20, slow=50):
    n = len(c)
    ema_fast = pd.Series(c).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(c).ewm(span=slow, adjust=False).mean().values
    entries = np.zeros(n, bool)
    for i in range(slow + 2, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            entries[i] = True
    return entries

def bollinger_entries(c, h, l, o, period=20, std=2):
    n = len(c)
    sma = pd.Series(c).rolling(period).mean().values
    std_dev = pd.Series(c).rolling(period).std().values
    entries = np.zeros(n, bool)
    for i in range(period + 1, n):
        if np.isnan(sma[i]) or np.isnan(std_dev[i]): continue
        upper = sma[i] + std * std_dev[i]
        if c[i] > upper and c[i-1] <= sma[i-1] + std * std_dev[i-1]:
            entries[i] = True
    return entries

def rsi_bounce_entries(c, h, l, o, period=14, threshold=30):
    n = len(c)
    if n < period + 2: return np.zeros(n, bool)
    delta = np.diff(c)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = np.full(n, np.nan); avg_loss = np.full(n, np.nan)
    for i in range(period, n):
        avg_gain[i] = np.mean(gain[i-period:i])
        avg_loss[i] = np.mean(loss[i-period:i])
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] == 0: rsi[i] = 100
        else: rsi[i] = 100 - 100 / (1 + avg_gain[i] / avg_loss[i])
    entries = np.zeros(n, bool)
    for i in range(period + 2, n):
        if np.isnan(rsi[i-1]) or np.isnan(rsi[i]): continue
        if rsi[i-1] < threshold and rsi[i] > threshold:
            entries[i] = True
    return entries

def macd_entries(c, h, l, o, fast=12, slow=26, signal=9):
    n = len(c)
    ema_fast = pd.Series(c).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(c).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    entries = np.zeros(n, bool)
    for i in range(slow + signal + 2, n):
        if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
            entries[i] = True
    return entries

# Main
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Load all coin data once
coin_data = {}
for sym in tradeable:
    data = load_last_month(sym)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    coin_data[sym] = resampled

strategies = {
    'Ichimoku 3/9/18': ichimoku_entries,
    'Breakout 20': breakout_entries,
    'EMA 20/50': ema_cross_entries,
    'Bollinger 20/2': bollinger_entries,
    'RSI <30': rsi_bounce_entries,
    'MACD 12/26/9': macd_entries,
}

print(f"🎯 اختبار استراتيجيات — يوليو 2026 | صفقتين × $500\n")
print(f"{'استراتيجية':>18s} | {'عملات':>4s} | {'إشارات':>5s} | {'منفذ':>4s} | {'WR':>5s} | {'ربح$':>7s} | {'سحب':>6s}")
print("─" * 70)

for name, entry_fn in strategies.items():
    coin_trades = {}
    for sym, (c8, h8, l8, o8, idx) in coin_data.items():
        entries = entry_fn(c8, h8, l8, o8)
        trades = run_trades(entries, None, c8, h8, l8, o8, idx)
        if len(trades) >= 1:
            coin_trades[sym] = trades
    
    if not coin_trades: continue
    avail = sum(len(v) for v in coin_trades.values())
    m = run_2positions(coin_trades)
    print(f"{name:>18s} | {len(coin_trades):4d} | {avail:5d} | {m['trades']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+6,.0f} | {m['dd']:5.1f}%")

# Now test on top 60 from 3-year for the best strategies
print(f"\n{'='*70}")
print(f"📊 باستخدام أفضل 60 عملة من 3 سنوات:")

def load(sym, period):
    p = os.path.join(f'/data/trading28/data/whale_15m_{period}', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

# Get 3-year top 60
coin_3y = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        entries = ichimoku_entries(c8, h8, l8, o8)
        trades = run_trades(entries, None, c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            coin_3y[sym] = coin_3y.get(sym, 0) + sum(p for _,_,p in trades)

top60_3y = set(c for c, _ in sorted(coin_3y.items(), key=lambda x: x[1], reverse=True)[:60])

for name, entry_fn in strategies.items():
    coin_trades = {}
    for sym in top60_3y & set(coin_data.keys()):
        c8, h8, l8, o8, idx = coin_data[sym]
        entries = entry_fn(c8, h8, l8, o8)
        trades = run_trades(entries, None, c8, h8, l8, o8, idx)
        if len(trades) >= 1:
            coin_trades[sym] = trades
    
    if not coin_trades: continue
    avail = sum(len(v) for v in coin_trades.values())
    m = run_2positions(coin_trades)
    print(f"{name:>18s} | {len(coin_trades):4d} | {avail:5d} | {m['trades']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+6,.0f} | {m['dd']:5.1f}%")
