#!/usr/bin/env python3
"""
QQE+SSL+EMA — 1h OPTIMIZATION — 3 years FET/USDT
Focus on reverse exit (the only profitable config)
Vary: QQE params, BB, threshold, SSL, EMA
"""
import ccxt, pandas as pd, numpy as np, sys, itertools, time
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 1095; CAP = 1000

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
            pass  # trend handled separately
    return smoothed_rsi.values, long_band, short_band

def simulate(c, h, l, long_entry, short_entry, warmup=200):
    n = len(c)
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, n):
        if pos == 0:
            if long_entry[i]: pos=1; ep=c[i]
            elif short_entry[i]: pos=-1; ep=c[i]
        elif pos == 1:
            if short_entry[i]:
                pnl = (c[i]/ep-1)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=-1; ep=c[i]
        elif pos == -1:
            if long_entry[i]:
                pnl = (1-c[i]/ep)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=1; ep=c[i]
        curve.append(eq)
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def metrics(trades, curve):
    if not trades or len(trades) < 5: return None
    pnls = trades; n = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    eq_s = pd.Series(curve)
    dds = ((eq_s-eq_s.expanding().max())/eq_s.expanding().max()*100).min()
    dr = eq_s.pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    fe = curve[-1]; ann = (fe/CAP)**(365/DAYS)-1
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

# ═══════════ FETCH ═══════════
print("Fetching FET/USDT 1h (3 years)...", flush=True)
df = fetch('1h', DAYS)
print(f"  {len(df)} candles", flush=True)
c = df['close'].values; h = df['high'].values; l = df['low'].values
n = len(c); warmup = 200

# ═══════════ PRE-COMPUTE QQE ═══════════
print("Pre-computing QQE variants...", flush=True)
QQE_CACHE = {}
RSI_LENS = [6, 9, 14]
SMOOTHS = [3, 5, 7]
FACTORS = [2.0, 3.0, 4.0, 5.0]

for rsi_len in RSI_LENS:
    for smooth in SMOOTHS:
        for factor in FACTORS:
            key = (rsi_len, smooth, factor)
            t0 = time.time()
            primary_rsi, _, _ = compute_qqe(df['close'], rsi_len, smooth, factor)
            QQE_CACHE[key] = primary_rsi
            print(f"  R{rsi_len} S{smooth} F{factor}: {time.time()-t0:.1f}s", flush=True)

# Secondary QQE (fixed: R6 S5 F1.61)
print("Secondary QQE...", flush=True)
secondary_rsi, _, _ = compute_qqe(df['close'], 6, 5, 1.61)

# SSL variants
print("Pre-computing SSL variants...", flush=True)
SSL_CACHE = {}
for ssl_exit_len in [10, 15, 20]:
    exit_high = hma(df['high'], ssl_exit_len).values
    exit_low = hma(df['low'], ssl_exit_len).values
    hlv3 = np.zeros(n); ssl_exit_val = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(exit_high[i]): hlv3[i] = hlv3[i-1]
        elif c[i] > exit_high[i]: hlv3[i] = 1
        elif c[i] < exit_low[i]: hlv3[i] = -1
        else: hlv3[i] = hlv3[i-1]
        ssl_exit_val[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
    ssl_bull = np.zeros(n, dtype=bool); ssl_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if not np.isnan(ssl_exit_val[i]):
            ssl_bull[i] = c[i] > ssl_exit_val[i] and c[i-1] <= ssl_exit_val[i-1]
            ssl_bear[i] = c[i] < ssl_exit_val[i] and c[i-1] >= ssl_exit_val[i-1]
    SSL_CACHE[ssl_exit_len] = (ssl_bull, ssl_bear)
    print(f"  SSL exit_len={ssl_exit_len}: done", flush=True)

# EMA variants
EMAS = {e: ema(df['close'], e).values for e in [100, 200]}

# ═══════════ GRID SEARCH ═══════════
print("\nGrid search...", flush=True)
BB_LENS = [30, 50]
BB_MULTS = [0.25, 0.35, 0.50]
THRESHOLDS = [2.0, 3.0, 4.0]

all_res = []
total = len(QQE_CACHE) * len(BB_LENS) * len(BB_MULTS) * len(THRESHOLDS) * len(SSL_CACHE) * len(EMAS)
done = 0

secondary_zero = secondary_rsi - 50

for (rsi1, smooth1, f1), primary_rsi in QQE_CACHE.items():
    primary_zero = primary_rsi - 50
    for bb_len in BB_LENS:
        bb_basis_raw = pd.Series(primary_zero).rolling(bb_len).mean().values
        bb_std_raw = pd.Series(primary_zero).rolling(bb_len).std().values
        for bb_mult in BB_MULTS:
            bb_upper = bb_basis_raw + bb_mult * bb_std_raw
            bb_lower = bb_basis_raw - bb_mult * bb_std_raw
            for threshold in THRESHOLDS:
                qqe_blue = (secondary_zero > threshold) & (primary_zero > bb_upper)
                qqe_red = (secondary_zero < -threshold) & (primary_zero < bb_lower)
                
                for ssl_len, (ssl_bull, ssl_bear) in SSL_CACHE.items():
                    for ema_len, ema_line in EMAS.items():
                        done += 1
                        if done % 100 == 0: print(f"  {done}/{total}...", flush=True)
                        
                        long_entry = np.zeros(n, dtype=bool)
                        short_entry = np.zeros(n, dtype=bool)
                        for i in range(warmup, n):
                            if np.isnan(ema_line[i]): continue
                            if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]:
                                long_entry[i] = True
                            elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]:
                                short_entry[i] = True
                        
                        n_sig = long_entry.sum() + short_entry.sum()
                        if n_sig < 5: continue
                        
                        trades, curve = simulate(c, h, l, long_entry, short_entry)
                        m = metrics(trades, curve)
                        if m:
                            m['label'] = f"R{rsi1} S{smooth1} F{f1} BB{bb_len}x{bb_mult} T{threshold} SSL{ssl_len} E{ema_len}"
                            m['sig'] = n_sig; all_res.append(m)

print(f"\n{'='*90}")
print(f"1h REVERSE OPTIMIZATION — {len(all_res)} valid configs")
print(f"{'='*90}")

# Rank by best balance: high WR + positive return
print("\n🏆 TOP 15 by Win Rate (min 20 trades):")
valid = [r for r in all_res if r['n'] >= 20]
for i, r in enumerate(sorted(valid, key=lambda x: x['wr'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")

print("\n🏆 TOP 15 by Return (min 20 trades):")
for i, r in enumerate(sorted(valid, key=lambda x: x['eq'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f} | Ann {r['annual']:>+6.1f}%")

print("\n🏆 TOP 15 best balance (WR>35% + positive return):")
balanced = [r for r in all_res if r['wr'] > 35 and r['eq'] > 1000 and r['n'] >= 10]
for i, r in enumerate(sorted(balanced, key=lambda x: (x['wr'], x['eq']), reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")
