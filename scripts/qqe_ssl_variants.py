#!/usr/bin/env python3
"""
QQE+SSL variants — 5 coins × 6 configs — 3 years 1h
Test fundamentally different approaches
"""
import ccxt, pandas as pd, numpy as np, sys, time, gc
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

COMM = 0.002; DAYS = 1095; CAP = 1000

COINS = ['FET/USDT','XRP/USDT','ATOM/USDT','MATIC/USDT','SOL/USDT']

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

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

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
        else: long_band[i] = new_long
        if not np.isnan(short_band[i-1]) and smoothed_rsi.iloc[i-1] < short_band[i-1] and smoothed_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short)
        else: short_band[i] = new_short
    return smoothed_rsi.values

def compute_signals(df, qqe_rsi, qqe_smooth, qqe_f, bb_len, bb_mult, threshold, ssl_exit_len, ema_len, use_ema=True):
    c = df['close'].values; h = df['high'].values; l = df['low'].values; n = len(c)
    
    primary_rsi = compute_qqe(df['close'], qqe_rsi, qqe_smooth, qqe_f)
    secondary_rsi = compute_qqe(df['close'], 6, 5, 1.61)
    
    primary_zero = primary_rsi - 50
    secondary_zero = secondary_rsi - 50
    
    bb_basis = pd.Series(primary_zero).rolling(bb_len).mean().values
    bb_std = pd.Series(primary_zero).rolling(bb_len).std().values
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    qqe_blue = (secondary_zero > threshold) & (primary_zero > bb_upper)
    qqe_red = (secondary_zero < -threshold) & (primary_zero < bb_lower)
    
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
    
    ema_line = ema(df['close'], ema_len).values if use_ema else np.ones(n)
    
    warmup = 200
    long_entry = np.zeros(n, dtype=bool); short_entry = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_line[i]): continue
        if use_ema:
            if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]: long_entry[i] = True
            elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]: short_entry[i] = True
        else:
            if qqe_blue[i] and ssl_bull[i]: long_entry[i] = True
            elif qqe_red[i] and ssl_bear[i]: short_entry[i] = True
    
    return long_entry, short_entry, c, h, l

def simulate(c, h, l, long_entry, short_entry, exit_mode, tp=None, sl=None, trail=None):
    n = len(c); warmup = 200
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; peak = 0
    
    for i in range(warmup, n):
        if pos == 1:
            exit_now = False; exit_px = c[i]
            if exit_mode == 'reverse':
                if short_entry[i]: exit_now = True
            elif exit_mode == 'tp_sl':
                if h[i] >= ep*(1+tp/100): exit_now = True; exit_px = ep*(1+tp/100)
                elif c[i] <= ep*(1-sl/100): exit_now = True; exit_px = c[i]
                elif short_entry[i]: exit_now = True
            elif exit_mode == 'trail':
                peak = max(peak, h[i])
                if c[i] <= peak*(1-trail/100): exit_now = True; exit_px = c[i]
                elif short_entry[i]: exit_now = True
            if exit_now:
                pnl = (exit_px/ep-1)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=0; peak=0
                if exit_mode == 'reverse' and short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
        elif pos == -1:
            exit_now = False; exit_px = c[i]
            if exit_mode == 'reverse':
                if long_entry[i]: exit_now = True
            elif exit_mode == 'tp_sl':
                if l[i] <= ep*(1-tp/100): exit_now = True; exit_px = ep*(1-tp/100)
                elif c[i] >= ep*(1+sl/100): exit_now = True; exit_px = c[i]
                elif long_entry[i]: exit_now = True
            elif exit_mode == 'trail':
                peak = min(peak, l[i]) if peak != 0 else l[i]
                if c[i] >= peak*(1+trail/100): exit_now = True; exit_px = c[i]
                elif long_entry[i]: exit_now = True
            if exit_now:
                pnl = (1-exit_px/ep)*100-COMM*100; trades.append(pnl)
                eq*=(1+pnl/100); pos=0; peak=0
                if exit_mode == 'reverse' and long_entry[i]: pos=1; ep=c[i]; peak=h[i]
        if pos == 0:
            if long_entry[i]: pos=1; ep=c[i]; peak=h[i]
            elif short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
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
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    return {'n':n,'wr':wr,'eq':curve[-1],'dd':dds,'sh':sh,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

# ═══════════ CONFIGS ═══════════
CONFIGS = [
    # (name, qqe_rsi, qqe_smooth, qqe_f, bb_len, bb_mult, threshold, ssl_exit, ema_len, use_ema, exit_mode, tp, sl, trail)
    ("A-Original REV",         6, 5, 2.0, 30, 0.5, 2.0, 10, 200, True,  'reverse', None,None,None),
    ("B-No EMA REV",           6, 5, 2.0, 30, 0.5, 2.0, 10, 200, False, 'reverse', None,None,None),
    ("C-Tight QQE REV",        3, 3, 4.0, 50, 0.35,3.0, 15, 200, True,  'reverse', None,None,None),
    ("D-Original TP3/SL2",     6, 5, 2.0, 30, 0.5, 2.0, 10, 200, True,  'tp_sl',   3.0,2.0,None),
    ("E-Tight QQE TP5/SL2",    3, 3, 4.0, 50, 0.35,3.0, 15, 200, True,  'tp_sl',   5.0,2.0,None),
    ("F-Strict EMA50 REV",     9, 7, 2.0, 50, 0.25,4.0, 20, 50,  True,  'reverse', None,None,None),
    ("G-LowSmooth REV",        14,3, 3.0, 30, 0.5, 2.0, 10, 100, True,  'reverse', None,None,None),
    ("H-WideBand Trail",       6, 5, 2.0, 30, 0.5, 2.0, 10, 200, True,  'trail',   None,None,0.5),
]

# ═══════════ RUN ═══════════
print("Testing 5 coins × 8 configs...\n")
all_results = []

for symbol in COINS:
    sym = symbol.replace('/USDT','')
    print(f"─── {sym} ───")
    df = fetch(symbol, '1h', DAYS)
    if df is None or len(df) < 2000:
        print(f"  SKIP")
        continue
    
    for cfg in CONFIGS:
        name = cfg[0]
        long_entry, short_entry, c, h, l = compute_signals(df, *cfg[1:10])
        trades, curve = simulate(c, h, l, long_entry, short_entry, cfg[10], cfg[11], cfg[12], cfg[13])
        m = metrics(trades, curve)
        if m:
            m['symbol'] = sym; m['config'] = name
            m['sig'] = long_entry.sum() + short_entry.sum()
            all_results.append(m)
        print(f"  {name:<22} {m['n']:>4d}t | WR {m['wr']:>5.1f}% | R:R {m['rr']:.2f}x | DD {m['dd']:>6.1f}% | ${m['eq']-1000:>+8.0f}" if m else f"  {name:<22} NO TRADES")
    
    del df; gc.collect()

# ═══════════ SUMMARY ═══════════
print(f"\n{'='*85}")
print(f"SUMMARY — Average across 5 coins")
print(f"{'='*85}")
print(f"{'Config':<25} {'Avg N':>6} {'Avg WR':>7} {'Avg R:R':>6} {'Avg DD':>7} {'Avg $':>8} {'Winners':>7}")
print("-"*75)

for cfg in CONFIGS:
    name = cfg[0]
    items = [r for r in all_results if r['config'] == name]
    if not items: continue
    avg_n = np.mean([r['n'] for r in items])
    avg_wr = np.mean([r['wr'] for r in items])
    avg_rr = np.mean([r['rr'] for r in items])
    avg_dd = np.mean([r['dd'] for r in items])
    avg_eq = np.mean([r['eq'] for r in items]) - 1000
    wins = sum(1 for r in items if r['eq'] > 1000)
    print(f"{name:<25} {avg_n:>6.0f} {avg_wr:>6.1f}% {avg_rr:>5.2f}x {avg_dd:>6.1f}% ${avg_eq:>+7.0f} {wins:>3}/{len(items)}")
