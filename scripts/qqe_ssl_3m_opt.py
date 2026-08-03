#!/usr/bin/env python3
"""
QQE+SSL+EMA — 3m Optimized Grid (pre-compute QQE once)
"""
import ccxt, pandas as pd, numpy as np, sys, itertools, time
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(SYMBOL, tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def rsi_s(s, p):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/p, adjust=False).mean()/l.ewm(alpha=1/p, adjust=False).mean())

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def hma(s, l):
    half = int(max(l/2, 2)); sq = int(max(np.sqrt(l), 1))
    w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1,half+1)), raw=True)
    w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1,l+1)), raw=True)
    return (2*w1 - w2).rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

def compute_qqe(close, rsi_len, smooth, factor):
    wilders_len = rsi_len * 2 - 1
    rsi_val = rsi_s(close, rsi_len)
    smoothed_rsi = ema(rsi_val, smooth)
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = ema(atr_rsi, wilders_len)
    dynamic_atr = smoothed_atr_rsi * factor
    n = len(close)
    long_band = np.full(n, np.nan); short_band = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    warm = max(wilders_len + 10, 50)
    for i in range(warm, n):
        new_short = smoothed_rsi.iloc[i] + dynamic_atr.iloc[i]
        new_long = smoothed_rsi.iloc[i] - dynamic_atr.iloc[i]
        if not np.isnan(long_band[i-1]) and smoothed_rsi.iloc[i-1] > long_band[i-1] and smoothed_rsi.iloc[i] > long_band[i-1]:
            long_band[i] = max(long_band[i-1], new_long)
        else:
            long_band[i] = new_long
        if not np.isnan(short_band[i-1]) and smoothed_rsi.iloc[i-1] < short_band[i-1] and smoothed_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short)
        else:
            short_band[i] = new_short
        if smoothed_rsi.iloc[i] > short_band[i-1] and smoothed_rsi.iloc[i-1] <= short_band[i-1]:
            trend[i] = 1
        elif smoothed_rsi.iloc[i] < long_band[i-1] and smoothed_rsi.iloc[i-1] >= long_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    return smoothed_rsi.values, long_band, short_band

def simulate(trades_list, exit_mode, tp, sl, trail):
    """Simulate with pre-computed signals"""
    c, h, l, long_entry, short_entry = [np.array(x) for x in trades_list]
    n = len(c); warmup = 200
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; peak = 0
    
    for i in range(warmup, n):
        if pos == 1:
            exit_now = False; exit_px = c[i]; reason = ''
            if exit_mode == 'reverse':
                if short_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'tp_sl':
                if h[i] >= ep*(1+tp/100): exit_now = True; exit_px = ep*(1+tp/100); reason = 'TP'
                elif c[i] <= ep*(1-sl/100): exit_now = True; exit_px = c[i]; reason = 'SL'
                elif short_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'trail':
                peak = max(peak, h[i])
                if c[i] <= peak*(1-trail/100): exit_now = True; exit_px = c[i]; reason = 'TRAIL'
                elif short_entry[i]: exit_now = True; reason = 'REV'
            if exit_now:
                pnl = (exit_px/ep-1)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=0; peak=0
                if reason == 'REV' and short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
        elif pos == -1:
            exit_now = False; exit_px = c[i]; reason = ''
            if exit_mode == 'reverse':
                if long_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'tp_sl':
                if l[i] <= ep*(1-tp/100): exit_now = True; exit_px = ep*(1-tp/100); reason = 'TP'
                elif c[i] >= ep*(1+sl/100): exit_now = True; exit_px = c[i]; reason = 'SL'
                elif long_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'trail':
                peak = min(peak, l[i]) if peak != 0 else l[i]
                if c[i] >= peak*(1+trail/100): exit_now = True; exit_px = c[i]; reason = 'TRAIL'
                elif long_entry[i]: exit_now = True; reason = 'REV'
            if exit_now:
                pnl = (1-exit_px/ep)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=0; peak=0
                if reason == 'REV' and long_entry[i]: pos=1; ep=c[i]; peak=h[i]
        if pos == 0:
            if long_entry[i]: pos=1; ep=c[i]; peak=h[i]
            elif short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
        curve.append(eq)
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def metrics(trades, curve):
    if not trades: return None
    pnls = trades; n = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    fe = curve[-1]; ann = (fe/CAP)**(365/DAYS)-1
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

# ═══════════ FETCH ═══════════
print("Fetching FET/USDT 3m...", flush=True)
df = fetch('3m', 180)
print(f"  {len(df)} candles", flush=True)

c = df['close'].values; h = df['high'].values; l = df['low'].values
n = len(c); warmup = 200

# ═══════════ PRE-COMPUTE QQE ═══════════
print("Pre-computing QQE variants...", flush=True)
QQE_CACHE = {}  # key: (rsi_len, factor) → primary_rsi
RSI1s = [3, 6, 9, 14]
F1s = [2.0, 3.0, 4.0, 5.0]
for rsi_len in RSI1s:
    for f1 in F1s:
        key = (rsi_len, f1)
        t0 = time.time()
        primary_rsi, _, _ = compute_qqe(df['close'], rsi_len, 5, f1)
        QQE_CACHE[key] = primary_rsi
        print(f"  QQE RSI={rsi_len} F={f1}: {time.time()-t0:.1f}s", flush=True)

# Pre-compute secondary QQE (fixed: RSI=6, F=1.61)
print("Pre-computing secondary QQE...", flush=True)
secondary_rsi, _, _ = compute_qqe(df['close'], 6, 5, 1.61)

# Pre-compute SSL (fixed: exit_len=15)
print("Pre-computing SSL...", flush=True)
ssl_exit_high = hma(df['high'], 15).values
ssl_exit_low = hma(df['low'], 15).values
ssl_hlv = np.zeros(n)
ssl_exit_val = np.full(n, np.nan)
for i in range(1, n):
    if np.isnan(ssl_exit_high[i]): ssl_hlv[i] = ssl_hlv[i-1]
    elif c[i] > ssl_exit_high[i]: ssl_hlv[i] = 1
    elif c[i] < ssl_exit_low[i]: ssl_hlv[i] = -1
    else: ssl_hlv[i] = ssl_hlv[i-1]
    ssl_exit_val[i] = ssl_exit_high[i] if ssl_hlv[i] < 0 else ssl_exit_low[i]
ssl_bull = np.zeros(n, dtype=bool); ssl_bear = np.zeros(n, dtype=bool)
for i in range(2, n):
    if not np.isnan(ssl_exit_val[i]):
        ssl_bull[i] = c[i] > ssl_exit_val[i] and c[i-1] <= ssl_exit_val[i-1]
        ssl_bear[i] = c[i] < ssl_exit_val[i] and c[i-1] >= ssl_exit_val[i-1]

# Pre-compute EMA variants
EMAS = {e: ema(df['close'], e).values for e in [50, 100, 200]}

# ═══════════ GRID SEARCH ═══════════
print("\nRunning grid search...", flush=True)
BB_LENS = [30, 50]
SSL_EXIT_LENS = [10, 15]
EMAS_L = [50, 100, 200]
EXITS = [
    ('reverse', None, None, None, 'REV'),
    ('tp_sl', 1.0, 0.5, None, 'TP1/SL0.5'),
    ('tp_sl', 1.5, 0.7, None, 'TP1.5/SL0.7'),
    ('trail', None, None, 0.05, 'TR0.05'),
]

all_res = []
total = len(QQE_CACHE) * len(BB_LENS) * len(EMAS_L) * len(EXITS)
done = 0

for (rsi1, f1), primary_rsi in QQE_CACHE.items():
    primary_zero = primary_rsi - 50
    secondary_zero = secondary_rsi - 50
    
    for bb_len in BB_LENS:
        # Bollinger
        bb_series = pd.Series(primary_zero, index=df.index)
        bb_basis = bb_series.rolling(bb_len).mean().values
        bb_std = bb_series.rolling(bb_len).std().values
        bb_upper = bb_basis + 0.35 * bb_std
        bb_lower = bb_basis - 0.35 * bb_std
        
        # QQE signals
        qqe_blue = (secondary_zero > 3.0) & (primary_zero > bb_upper)
        qqe_red = (secondary_zero < -3.0) & (primary_zero < bb_lower)
        
        for ema_len in EMAS_L:
            ema_line = EMAS[ema_len]
            
            # Entry signals
            long_entry = np.zeros(n, dtype=bool)
            short_entry = np.zeros(n, dtype=bool)
            for i in range(warmup, n):
                if np.isnan(ema_line[i]): continue
                if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]:
                    long_entry[i] = True
                elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]:
                    short_entry[i] = True
            
            n_long = long_entry.sum(); n_short = short_entry.sum()
            if n_long + n_short < 3: continue  # skip dead configs
            
            for exit_mode, tp, sl, trail, exit_label in EXITS:
                done += 1
                if done % 50 == 0: print(f"  {done}/{total}...", flush=True)
                
                trades, curve = simulate([c, h, l, long_entry, short_entry], exit_mode, tp, sl, trail)
                m = metrics(trades, curve)
                if m and m['n'] >= 3:
                    m['label'] = f"RSI{rsi1} F{f1} BB{bb_len} E{ema_len} {exit_label}"
                    m['n_long'] = n_long; m['n_short'] = n_short
                    all_res.append(m)

# ═══════════ RESULTS ═══════════
print(f"\n{'='*90}")
print(f"QQE+SSL+EMA — 3m FET/USDT — {DAYS} days — {len(all_res)} valid configs")
print(f"{'='*90}")

print("\n🏆 TOP 15 by Win Rate:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['wr'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<45} | {r['n']:>4d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")

print("\n🏆 TOP 15 by Return:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['eq'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<45} | {r['n']:>4d}t | WR {r['wr']:>5.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f}")

print("\n🏆 TOP 15 by Sharpe:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['sh'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<45} | {r['n']:>4d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | Sharpe {r['sh']:>5.2f} | ${r['eq']-1000:>+8.0f}")
