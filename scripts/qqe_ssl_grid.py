#!/usr/bin/env python3
"""
QQE + SSL + EMA Grid Search — FET/USDT
Vary: EMA length, Timeframe, Exit type
"""
import ccxt, pandas as pd, numpy as np, sys, itertools
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(tf):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=DAYS)).isoformat())
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
    half = int(l/2); sq = int(np.sqrt(l))
    w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1,half+1)), raw=True)
    w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1,l+1)), raw=True)
    diff = 2*w1 - w2
    return diff.rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

def compute_qqe(close, rsi_len=6, smooth=5, factor=3.0):
    wilders_len = rsi_len * 2 - 1
    rsi_val = rsi_s(close, rsi_len)
    smoothed_rsi = ema(rsi_val, smooth)
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = ema(atr_rsi, wilders_len)
    dynamic_atr = smoothed_atr_rsi * factor
    n = len(close)
    long_band = np.full(n, np.nan); short_band = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    for i in range(wilders_len + 10, n):
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

def compute_signals(df):
    c = df['close']
    primary_rsi, primary_trend = compute_qqe(c, 6, 5, 3.0)
    secondary_rsi, secondary_trend = compute_qqe(c, 6, 5, 1.61)
    
    # Primary QQE Bollinger (zero-centered)
    primary_zero = pd.Series(primary_rsi - 50, index=df.index)
    bb_basis = primary_zero.rolling(50).mean()
    bb_std = primary_zero.rolling(50).std()
    bb_upper = bb_basis + 0.35 * bb_std
    bb_lower = bb_basis - 0.35 * bb_std
    
    secondary_zero = pd.Series(secondary_rsi - 50, index=df.index)
    primary_rsi_zero = pd.Series(primary_rsi - 50, index=df.index)
    threshold = 3.0
    
    qqe_blue = (secondary_zero > threshold) & (primary_rsi_zero > bb_upper)
    qqe_red = (secondary_zero < -threshold) & (primary_rsi_zero < bb_lower)
    
    # SSL Exit signals
    exit_high = hma(df['high'], 15).values
    exit_low = hma(df['low'], 15).values
    n = len(c)
    cl = c.values
    hlv3 = np.zeros(n)
    ssl_exit = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(exit_high[i]): hlv3[i] = hlv3[i-1]
        elif cl[i] > exit_high[i]: hlv3[i] = 1
        elif cl[i] < exit_low[i]: hlv3[i] = -1
        else: hlv3[i] = hlv3[i-1]
        ssl_exit[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
    
    ssl_bull = np.zeros(n, dtype=bool); ssl_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if np.isnan(ssl_exit[i]): continue
        ssl_bull[i] = cl[i] > ssl_exit[i] and cl[i-1] <= ssl_exit[i-1]
        ssl_bear[i] = cl[i] < ssl_exit[i] and cl[i-1] >= ssl_exit[i-1]
    
    return qqe_blue.values, qqe_red.values, ssl_bull, ssl_bear

# ═══════════ EXIT TYPES ═══════════
def exit_type(name):
    return name

def backtest_with_exit(df, ema_len, exit_mode, tp_pct=None, sl_pct=None, trail_pct=None):
    warmup = 200
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    ema_line = ema(df['close'], ema_len).values
    qqe_blue, qqe_red, ssl_bull, ssl_bear = compute_signals(df)
    n = len(c)
    
    # Entry conditions
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_line[i]): continue
        if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]:
            long_entry[i] = True
        elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]:
            short_entry[i] = True
    
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; entry_i = 0
    peak = 0  # for trailing
    
    for i in range(warmup, n):
        # Check exit conditions
        if pos == 1:
            exit_now = False; exit_px = c[i]; exit_reason = ''
            
            if exit_mode == 'reverse':
                if short_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'tp_sl':
                tp = ep * (1 + tp_pct/100)
                sl = ep * (1 - sl_pct/100)
                if h[i] >= tp:
                    exit_now = True; exit_px = tp; exit_reason = 'TP'
                elif c[i] <= sl:
                    exit_now = True; exit_px = c[i]; exit_reason = 'SL'
                elif short_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'trail':
                peak = max(peak, h[i])
                trail_price = peak * (1 - trail_pct/100)
                if c[i] <= trail_price:
                    exit_now = True; exit_px = c[i]; exit_reason = 'TRAIL'
                elif short_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'tp_trail':
                peak = max(peak, h[i])
                tp = ep * (1 + tp_pct/100)
                trail_price = peak * (1 - trail_pct/100)
                if h[i] >= tp:
                    exit_now = True; exit_px = tp; exit_reason = 'TP'
                elif c[i] <= trail_price:
                    exit_now = True; exit_px = c[i]; exit_reason = 'TRAIL'
                elif short_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            if exit_now:
                pnl = (exit_px/ep-1)*100 - COMM*100
                trades.append({'pnl':pnl, 'exit':exit_reason})
                eq *= (1+pnl/100)
                pos = 0; peak = 0
                # Check if we should flip
                if exit_reason == 'REV' and short_entry[i]:
                    pos, ep, entry_i = -1, c[i], i
                    peak = l[i] if l[i] < peak or peak == 0 else peak
        
        elif pos == -1:
            exit_now = False; exit_px = c[i]; exit_reason = ''
            
            if exit_mode == 'reverse':
                if long_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'tp_sl':
                tp = ep * (1 - tp_pct/100)
                sl = ep * (1 + sl_pct/100)
                if l[i] <= tp:
                    exit_now = True; exit_px = tp; exit_reason = 'TP'
                elif c[i] >= sl:
                    exit_now = True; exit_px = c[i]; exit_reason = 'SL'
                elif long_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'trail':
                peak = min(peak, l[i]) if peak != 0 else l[i]
                trail_price = peak * (1 + trail_pct/100)
                if c[i] >= trail_price:
                    exit_now = True; exit_px = c[i]; exit_reason = 'TRAIL'
                elif long_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            elif exit_mode == 'tp_trail':
                peak = min(peak, l[i]) if peak != 0 else l[i]
                tp = ep * (1 - tp_pct/100)
                trail_price = peak * (1 + trail_pct/100)
                if l[i] <= tp:
                    exit_now = True; exit_px = tp; exit_reason = 'TP'
                elif c[i] >= trail_price:
                    exit_now = True; exit_px = c[i]; exit_reason = 'TRAIL'
                elif long_entry[i]:
                    exit_now = True; exit_reason = 'REV'
            
            if exit_now:
                pnl = (1-exit_px/ep)*100 - COMM*100
                trades.append({'pnl':pnl, 'exit':exit_reason})
                eq *= (1+pnl/100)
                pos = 0; peak = 0
                if exit_reason == 'REV' and long_entry[i]:
                    pos, ep, entry_i = 1, c[i], i
                    peak = h[i]
        
        # New entry if flat
        if pos == 0:
            if long_entry[i]:
                pos, ep, entry_i = 1, c[i], i
                peak = h[i]
            elif short_entry[i]:
                pos, ep, entry_i = -1, c[i], i
                peak = l[i]
        
        curve.append(eq)
    
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl, 'exit':'EOD'})
        eq *= (1+pnl/100); curve.append(eq)
    
    return trades, curve

def calc_metrics(trades, curve):
    if not trades: return None
    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    fe = curve[-1]; ann = (fe/CAP)**(365/DAYS)-1
    tp_hits = sum(1 for t in trades if t.get('exit')=='TP')
    sl_hits = sum(1 for t in trades if t.get('exit')=='SL')
    trail_hits = sum(1 for t in trades if t.get('exit')=='TRAIL')
    rev_hits = sum(1 for t in trades if t.get('exit')=='REV')
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l),
            'tp':tp_hits,'sl':sl_hits,'trail':trail_hits,'rev':rev_hits,'tp_sum':sum(w),'tl_sum':sum(l)}

# ═══════════ RUN GRID ═══════════
TFS = ['15m', '30m', '1h', '4h']
EMAS = [50, 100, 200]
EXITS = [
    ('reverse', None, None, None),
    ('tp_sl', 3.0, 1.5, None),
    ('tp_sl', 5.0, 2.0, None),
    ('trail', None, None, 0.5),
    ('tp_trail', 3.0, None, 0.3),
    ('tp_trail', 5.0, None, 0.5),
]

print("Fetching FET/USDT all TFs...")
data = {}
for tf in TFS:
    data[tf] = fetch(tf)
    print(f"  {tf}: {len(data[tf])} candles")

print(f"\n{'='*90}")
print(f"QQE+SSL+EMA Grid Search — FET/USDT — {DAYS} days")
print(f"{'='*90}")

all_res = []
for tf, ema_len, (exit_mode, tp, sl, trail) in itertools.product(TFS, EMAS, EXITS):
    df = data[tf]
    label = f"{tf} EMA{ema_len} {exit_mode}"
    if tp: label += f" TP{tp}"
    if sl: label += f" SL{sl}"
    if trail: label += f" TR{trail}"
    
    trades, curve = backtest_with_exit(df, ema_len, exit_mode, tp, sl, trail)
    m = calc_metrics(trades, curve)
    if m and m['n'] > 0:
        m['config'] = label; m['tf'] = tf; m['ema'] = ema_len; m['exit_mode'] = exit_mode
        all_res.append(m)

# ═══════════ RANKINGS ═══════════
print("\n🏆 TOP 15 by Win Rate:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['wr'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['config']:<40} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")

print("\n🏆 TOP 15 by Return:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['eq'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    e = f"🎯{r['tp']} 🛑{r['sl']} 🐌{r['trail']} 🔄{r['rev']}" if 'tp' in r else ''
    print(f"{icon:>3} {r['config']:<40} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f} | {e}")

print("\n🏆 TOP 15 by Sharpe:")
for i, r in enumerate(sorted(all_res, key=lambda x: x['sh'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['config']:<40} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | Sharpe {r['sh']:>5.2f} | ${r['eq']-1000:>+8.0f}")
