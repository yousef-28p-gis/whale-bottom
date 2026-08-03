#!/usr/bin/env python3
"""
QQE+SSL+EMA — Optimized config — 10 Altcoins — 3 years 1h
Config: R6 S5 F2.0 BB30×0.5 T2.0 SSL10 E200
"""
import ccxt, pandas as pd, numpy as np, sys, time, gc
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

COMM = 0.002; DAYS = 1095; CAP = 1000

COINS = ['ADA/USDT','XRP/USDT','SOL/USDT','LINK/USDT','DOT/USDT',
         'AVAX/USDT','MATIC/USDT','ATOM/USDT','ARB/USDT','APT/USDT']

def fetch(symbol, tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
            if not batch: break
            all_c.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 1000: break
        except: break
    if not all_c: return None
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

def compute_qqe(close, rsi_len=6, smooth=5, factor=2.0):
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
    return smoothed_rsi.values

def run_backtest(symbol):
    t0 = time.time()
    print(f"  {symbol}: fetching...", flush=True)
    df = fetch(symbol, '1h', DAYS)
    if df is None or len(df) < 2000:
        print(f"    SKIP: insufficient data ({len(df) if df is not None else 0})", flush=True)
        return None
    
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    n = len(c); warmup = 200
    
    # QQE primary + secondary
    primary_rsi = compute_qqe(df['close'], 6, 5, 2.0)
    secondary_rsi = compute_qqe(df['close'], 6, 5, 1.61)
    
    primary_zero = primary_rsi - 50
    secondary_zero = secondary_rsi - 50
    
    # BB30×0.5
    bb_basis = pd.Series(primary_zero).rolling(30).mean().values
    bb_std = pd.Series(primary_zero).rolling(30).std().values
    bb_upper = bb_basis + 0.5 * bb_std
    bb_lower = bb_basis - 0.5 * bb_std
    
    # QQE signals (T2.0)
    qqe_blue = (secondary_zero > 2.0) & (primary_zero > bb_upper)
    qqe_red = (secondary_zero < -2.0) & (primary_zero < bb_lower)
    
    # SSL10
    exit_high = hma(df['high'], 10).values
    exit_low = hma(df['low'], 10).values
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
    
    # EMA200
    ema_line = ema(df['close'], 200).values
    
    # Entry
    long_entry = np.zeros(n, dtype=bool); short_entry = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_line[i]): continue
        if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]: long_entry[i] = True
        elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]: short_entry[i] = True
    
    # Simulate
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
    
    del df; gc.collect()
    
    if not trades or len(trades) < 5: return None
    pnls = trades; nt = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/nt*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    ann = (eq/CAP)**(365/DAYS)-1
    print(f"    {nt}t | WR {wr:.1f}% | R:R {rr:.2f}x | DD {dds:.1f}% | ${eq-1000:+.0f} | {time.time()-t0:.1f}s", flush=True)
    return {'symbol':symbol.replace('/USDT',''),'n':nt,'wr':wr,'eq':eq,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l),'sig':long_entry.sum()+short_entry.sum()}

# ═══════════ RUN ═══════════
print(f"QQE+SSL+EMA Optimized — 10 Altcoins — 3 Years 1h")
print(f"Config: R6 S5 F2.0 BB30×0.5 T2.0 SSL10 E200\n")

results = []
for coin in COINS:
    r = run_backtest(coin)
    if r: results.append(r)

print(f"\n{'='*80}")
print(f"RESULTS — Sorted by Return")
print(f"{'='*80}")
print(f"{'Coin':<8} {'Trades':>6} {'WR':>7} {'R:R':>6} {'DD':>7} {'Equity':>10} {'Annual':>8} {'Sharpe':>6}")
print("-"*70)
for r in sorted(results, key=lambda x: x['eq'], reverse=True):
    print(f"{r['symbol']:<8} {r['n']:>6} {r['wr']:>6.1f}% {r['rr']:>5.2f}x {r['dd']:>6.1f}% ${r['eq']:>9.0f} {r['annual']:>+7.1f}% {r['sh']:>6.2f}")

# Summary
winners = [r for r in results if r['eq'] > CAP]
losers = [r for r in results if r['eq'] <= CAP]
print(f"\n✅ Winners: {len(winners)}/{len(results)} | ❌ Losers: {len(losers)}/{len(results)}")
if winners:
    avg_wr = np.mean([r['wr'] for r in winners])
    avg_ret = np.mean([(r['eq']/CAP-1)*100 for r in winners])
    print(f"   Avg WR: {avg_wr:.1f}% | Avg Return: +{avg_ret:.0f}%")
