#!/usr/bin/env python3
"""
QQE+SSL+EMA — 3m Grid Search — FET/USDT
Vary indicator parameters + exit types
"""
import ccxt, pandas as pd, numpy as np, sys, itertools
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
    diff = 2*w1 - w2
    return diff.rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

def compute_qqe_bands(close, rsi_len=6, smooth=5, factor=3.0):
    wilders_len = rsi_len * 2 - 1
    rsi_val = rsi_s(close, rsi_len)
    smoothed_rsi = ema(rsi_val, smooth)
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = ema(atr_rsi, wilders_len)
    dynamic_atr = smoothed_atr_rsi * factor
    n = len(close)
    long_band = np.full(n, np.nan); short_band = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    warm = wilders_len + 10
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
    return smoothed_rsi.values, trend

def backtest_config(df, qqe_rsi1, qqe_smooth1, qqe_f1, qqe_rsi2, qqe_smooth2, qqe_f2,
                    bb_len, bb_mult, threshold, ssl_baseline, ssl_exit_len, ema_len, exit_mode, tp, sl, trail):
    warmup = 200
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    n = len(c)
    
    # QQE
    primary_rsi, _ = compute_qqe_bands(df['close'], qqe_rsi1, qqe_smooth1, qqe_f1)
    secondary_rsi, _ = compute_qqe_bands(df['close'], qqe_rsi2, qqe_smooth2, qqe_f2)
    
    primary_zero = pd.Series(primary_rsi - 50, index=df.index)
    bb_basis = primary_zero.rolling(bb_len).mean()
    bb_std = primary_zero.rolling(bb_len).std()
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    secondary_zero = pd.Series(secondary_rsi - 50, index=df.index)
    
    qqe_blue = (secondary_zero > threshold) & (pd.Series(primary_rsi - 50, index=df.index) > bb_upper)
    qqe_red = (secondary_zero < -threshold) & (pd.Series(primary_rsi - 50, index=df.index) < bb_lower)
    
    # SSL
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
    
    # EMA
    ema_line = ema(df['close'], ema_len).values
    
    # Entry
    qb = qqe_blue.values; qr = qqe_red.values
    long_entry = np.zeros(n, dtype=bool); short_entry = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_line[i]): continue
        if qb[i] and ssl_bull[i] and c[i] > ema_line[i]: long_entry[i] = True
        elif qr[i] and ssl_bear[i] and c[i] < ema_line[i]: short_entry[i] = True
    
    # Simulation
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; peak = 0
    
    for i in range(warmup, n):
        if pos == 1:
            exit_now = False; exit_px = c[i]; reason = ''
            if exit_mode == 'reverse':
                if short_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'tp_sl':
                tpi = ep*(1+tp/100); sli = ep*(1-sl/100)
                if h[i] >= tpi: exit_now = True; exit_px = tpi; reason = 'TP'
                elif c[i] <= sli: exit_now = True; exit_px = c[i]; reason = 'SL'
                elif short_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'trail':
                peak = max(peak, h[i])
                if c[i] <= peak*(1-trail/100): exit_now = True; exit_px = c[i]; reason = 'TRAIL'
                elif short_entry[i]: exit_now = True; reason = 'REV'
            if exit_now:
                pnl = (exit_px/ep-1)*100-COMM*100; trades.append({'pnl':pnl,'exit':reason})
                eq*=(1+pnl/100); pos=0; peak=0
                if reason == 'REV' and short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
        elif pos == -1:
            exit_now = False; exit_px = c[i]; reason = ''
            if exit_mode == 'reverse':
                if long_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'tp_sl':
                tpi = ep*(1-tp/100); sli = ep*(1+sl/100)
                if l[i] <= tpi: exit_now = True; exit_px = tpi; reason = 'TP'
                elif c[i] >= sli: exit_now = True; exit_px = c[i]; reason = 'SL'
                elif long_entry[i]: exit_now = True; reason = 'REV'
            elif exit_mode == 'trail':
                peak = min(peak, l[i]) if peak != 0 else l[i]
                if c[i] >= peak*(1+trail/100): exit_now = True; exit_px = c[i]; reason = 'TRAIL'
                elif long_entry[i]: exit_now = True; reason = 'REV'
            if exit_now:
                pnl = (1-exit_px/ep)*100-COMM*100; trades.append({'pnl':pnl,'exit':reason})
                eq*=(1+pnl/100); pos=0; peak=0
                if reason == 'REV' and long_entry[i]: pos=1; ep=c[i]; peak=h[i]
        
        if pos == 0:
            if long_entry[i]: pos=1; ep=c[i]; peak=h[i]
            elif short_entry[i]: pos=-1; ep=c[i]; peak=l[i]
        curve.append(eq)
    
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl,'exit':'EOD'}); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def metrics(trades, curve):
    if not trades: return None
    pnls = [t['pnl'] for t in trades]; n = len(pnls)
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
print("Fetching FET/USDT 3m (180 days)...")
df = fetch('3m', 180)
print(f"  {len(df)} candles")

# ═══════════ GRID SEARCH ═══════════
QQE_RSI1 = [3, 6, 9]
QQE_F1 = [2.0, 3.0, 4.0]
BB_LEN = [30, 50, 70]
SSL_EXIT = [10, 15, 20]
EMA_LEN = [50, 100, 200]
EXITS = [
    ('reverse', None, None, None, 'REV'),
    ('tp_sl', 1.0, 0.5, None, 'TP1/SL0.5'),
    ('tp_sl', 1.5, 0.5, None, 'TP1.5/SL0.5'),
    ('trail', None, None, 0.08, 'TR0.08'),
]

# Fixed params (keep secondary QQE + threshold + baseline constant)
combos = list(itertools.product(QQE_RSI1, QQE_F1, BB_LEN, SSL_EXIT, EMA_LEN))
print(f"\nTesting {len(combos)} combinations × {len(EXITS)} exits = {len(combos)*len(EXITS)} total...")

all_res = []
for rsi1, f1, bb_len, ssl_exit_len, ema_len in combos:
    for exit_mode, tp, sl, trail, exit_label in EXITS:
        label = f"RSI{rsi1} F{f1} BB{bb_len} SSL{ssl_exit_len} EMA{ema_len} {exit_label}"
        trades, curve = backtest_config(df, rsi1, 5, f1, 6, 5, 1.61,
                                        bb_len, 0.35, 3.0, 60, ssl_exit_len, ema_len,
                                        exit_mode, tp, sl, trail)
        m = metrics(trades, curve)
        if m and m['n'] >= 3:
            m['label'] = label; m['rsi1'] = rsi1; m['f1'] = f1
            m['bb_len'] = bb_len; m['ssl'] = ssl_exit_len; m['ema'] = ema_len
            m['exit'] = exit_label
            all_res.append(m)

print(f"\n{'='*90}")
print(f"TOP RESULTS — 3m FET/USDT — {DAYS} days")
print(f"{'='*90}")

print("\n🏆 TOP 15 by Win Rate:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['wr'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")

print("\n🏆 TOP 15 by Return:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['eq'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f} | Ann {r['annual']:>+6.1f}%")

print("\n🏆 TOP 15 by Sharpe:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['sh'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['label']:<55} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | Sharpe {r['sh']:>5.2f} | ${r['eq']-1000:>+8.0f}")
