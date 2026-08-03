#!/usr/bin/env python3
"""
QQE+SSL+EMA — Tighter indicator settings — FET 1h 180d
Goal: fewer but higher quality signals
"""
import ccxt, pandas as pd, numpy as np, sys
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

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

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
    warm = max(wilders_len + 10, 50)
    for i in range(warm, n):
        new_short = smoothed_rsi.iloc[i] + dynamic_atr.iloc[i]
        new_long = smoothed_rsi.iloc[i] - dynamic_atr.iloc[i]
        if not np.isnan(long_band[i-1]) and smoothed_rsi.iloc[i-1] > long_band[i-1] and smoothed_rsi.iloc[i] > long_band[i-1]:
            long_band[i] = max(long_band[i-1], new_long)
        else: long_band[i] = new_long
        if not np.isnan(short_band[i-1]) and smoothed_rsi.iloc[i-1] < short_band[i-1] and smoothed_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short)
        else: short_band[i] = new_short
    return smoothed_rsi.values

def simulate(c, h, l, le, se):
    n = len(c); warmup = 200
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, n):
        if pos == 0:
            if le[i]: pos=1; ep=c[i]
            elif se[i]: pos=-1; ep=c[i]
        elif pos == 1:
            if se[i]:
                pnl = (c[i]/ep-1)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=-1; ep=c[i]
        elif pos == -1:
            if le[i]:
                pnl = (1-c[i]/ep)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=1; ep=c[i]
        curve.append(eq)
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def metrics(trades, curve):
    if not trades or len(trades) < 3: return None
    pnls = trades; n = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    return {'n':n,'wr':wr,'eq':curve[-1],'dd':dds,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

print("Fetching FET/USDT 1h...")
df = fetch('1h', DAYS)
c = df['close'].values; h = df['high'].values; l = df['low'].values
n = len(c); warmup = 200

# Pre-compute EMA variants
EMA200 = ema(df['close'], 200).values
EMA100 = ema(df['close'], 100).values
EMA50 = ema(df['close'], 50).values

# Pre-compute QQE variants
print("Pre-computing QQE variants...")
QQE_PRIMARY = {}
QQE_SECONDARY = {}

for rsi_len, smooth, factor in [(6,5,2.0),(6,5,3.0),(6,5,4.0),(9,5,2.0),(9,7,2.0),(9,7,3.0),(14,7,3.0),(6,3,3.0),(6,3,4.0)]:
    key = (rsi_len, smooth, factor)
    QQE_PRIMARY[key] = compute_qqe(df['close'], rsi_len, smooth, factor)
    print(f"  QQE R{rsi_len} S{smooth} F{factor}")

# Secondary fixed: 6,5,1.61
secondary_rsi = compute_qqe(df['close'], 6, 5, 1.61)
secondary_zero = secondary_rsi - 50

# Pre-compute SSL variants
print("Pre-computing SSL variants...")
SSL_CACHE = {}
for ssl_len in [10, 15, 20, 25]:
    exit_high = hma(df['high'], ssl_len).values
    exit_low = hma(df['low'], ssl_len).values
    hlv3 = np.zeros(n); ssl_exit_val = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(exit_high[i]): hlv3[i] = hlv3[i-1]
        elif c[i] > exit_high[i]: hlv3[i] = 1
        elif c[i] < exit_low[i]: hlv3[i] = -1
        else: hlv3[i] = hlv3[i-1]
        ssl_exit_val[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
    ssl_b = np.zeros(n, dtype=bool); ssl_be = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if not np.isnan(ssl_exit_val[i]):
            ssl_b[i] = c[i] > ssl_exit_val[i] and c[i-1] <= ssl_exit_val[i-1]
            ssl_be[i] = c[i] < ssl_exit_val[i] and c[i-1] >= ssl_exit_val[i-1]
    SSL_CACHE[ssl_len] = (ssl_b, ssl_be)
    print(f"  SSL exit={ssl_len}")

# ═══════════ CONFIGS ═══════════
# Each config: (name, qqe_key, bb_len, bb_mult, threshold, ssl_len, ema_array)
print("\nTesting configs...\n")
CONFIGS = [
    ("BASE R6F2 BB30x0.5 T2 S10 E200", (6,5,2.0), 30, 0.5, 2.0, 10, EMA200),
    ("TIGHT R6F4 BB50x0.35 T3 S15 E200", (6,5,4.0), 50, 0.35, 3.0, 15, EMA200),
    ("TIGHT R9F3 BB50x0.35 T4 S20 E200", (9,7,3.0), 50, 0.35, 4.0, 20, EMA200),
    ("SLOW R14F3 BB50x0.35 T3 S20 E200", (14,7,3.0), 50, 0.35, 3.0, 20, EMA200),
    ("FAST R6F2 BB30x0.3 T3 S10 E100", (6,5,2.0), 30, 0.3, 3.0, 10, EMA100),
    ("FAST R6F3 BB30x0.35 T3 S10 E100", (6,3,3.0), 30, 0.35, 3.0, 10, EMA100),
    ("WIDE R6F2 BB50x0.6 T2 S15 E200", (6,5,2.0), 50, 0.6, 2.0, 15, EMA200),
    ("ULTRA R6F4 BB50x0.5 T4 S25 E50", (6,5,4.0), 50, 0.5, 4.0, 25, EMA50),
    ("LOOSE R9F2 BB50x0.5 T2 S20 E50", (9,5,2.0), 50, 0.5, 2.0, 20, EMA50),
    ("EDGE R6F3 BB30x0.5 T3 S15 E200", (6,3,3.0), 30, 0.5, 3.0, 15, EMA200),
    ("HYBRID R9F2 BB30x0.5 T3 S15 E200", (9,5,2.0), 30, 0.5, 3.0, 15, EMA200),
    ("MAX R6F4 BB50x0.4 T3 S25 E100", (6,3,4.0), 50, 0.4, 3.0, 25, EMA100),
]

results = []

for name, qqe_key, bb_len, bb_mult, threshold, ssl_len, ema_arr in CONFIGS:
    primary_rsi = QQE_PRIMARY[qqe_key]
    primary_zero = primary_rsi - 50
    
    bb_basis = pd.Series(primary_zero).rolling(bb_len).mean().values
    bb_std = pd.Series(primary_zero).rolling(bb_len).std().values
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    qqe_blue = (secondary_zero > threshold) & (primary_zero > bb_upper)
    qqe_red = (secondary_zero < -threshold) & (primary_zero < bb_lower)
    
    ssl_b, ssl_be = SSL_CACHE[ssl_len]
    
    le = np.zeros(n, dtype=bool); se = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_arr[i]): continue
        if qqe_blue[i] and ssl_b[i] and c[i] > ema_arr[i]: le[i] = True
        elif qqe_red[i] and ssl_be[i] and c[i] < ema_arr[i]: se[i] = True
    
    sigs = le.sum() + se.sum()
    trades, curve = simulate(c, h, l, le, se)
    m = metrics(trades, curve)
    if m:
        icon = "✅" if m['eq'] > CAP else "🔴"
        results.append(m)
        print(f"{icon} {name:<45} | {m['n']:>3d}t ({sigs}s) | WR {m['wr']:>5.1f}% | R:R {m['rr']:.2f}x | DD {m['dd']:>6.1f}% | ${m['eq']-1000:>+8.0f}")
    else:
        print(f"⚪ {name:<45} | 0 trades ({sigs} signals)")

print(f"\n{'='*80}")
print("RANKED BY WR:")
for i, r in enumerate(sorted(results, key=lambda x: x['wr'], reverse=True)):
    print(f"  {i+1:>2}. WR {r['wr']:>5.1f}% | {r['n']:>3d}t | R:R {r['rr']:.2f}x | DD {r['dd']:>5.1f}% | ${r['eq']-1000:>+7.0f} | AW {r['aw']:>+5.1f}% | AL {r['al']:>+5.1f}%")

print(f"\n{'='*80}")
print("RANKED BY RETURN:")
for i, r in enumerate(sorted(results, key=lambda x: x['eq'], reverse=True)):
    print(f"  {i+1:>2}. ${r['eq']-1000:>+7.0f} | WR {r['wr']:>5.1f}% | {r['n']:>3d}t | R:R {r['rr']:.2f}x | DD {r['dd']:>5.1f}%")
