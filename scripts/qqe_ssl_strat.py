#!/usr/bin/env python3
"""
QQE MOD + SSL Hybrid + EMA200 — Combined Strategy
FET/USDT backtest across timeframes
"""
import ccxt, pandas as pd, numpy as np, sys
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

# ═══════════ QQE MOD ═══════════
def compute_qqe(close, rsi_len=6, smooth=5, factor=3.0):
    """Returns: qqe_trend_line, smoothed_rsi, trend_direction (1=bull, -1=bear)"""
    wilders_len = rsi_len * 2 - 1
    rsi_val = rsi_s(close, rsi_len)
    smoothed_rsi = ema(rsi_val, smooth)
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = ema(atr_rsi, wilders_len)
    dynamic_atr = smoothed_atr_rsi * factor
    
    n = len(close)
    long_band = np.full(n, np.nan)
    short_band = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    qqe_line = np.full(n, np.nan)
    
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
        
        # Cross detection
        if smoothed_rsi.iloc[i] > short_band[i-1] and smoothed_rsi.iloc[i-1] <= short_band[i-1]:
            trend[i] = 1
        elif smoothed_rsi.iloc[i] < long_band[i-1] and smoothed_rsi.iloc[i-1] >= long_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        qqe_line[i] = long_band[i] if trend[i] == 1 else short_band[i]
    
    return qqe_line, smoothed_rsi.values, trend

def compute_qqe_signals(df):
    """Returns: long_signal, short_signal boolean arrays"""
    c = df['close']
    
    # Primary QQE (factor=3.0)
    primary_line, primary_rsi, primary_trend = compute_qqe(c, 6, 5, 3.0)
    
    # Secondary QQE (factor=1.61)
    secondary_line, secondary_rsi, secondary_trend = compute_qqe(c, 6, 5, 1.61)
    
    # Bollinger on primary QQE line (zero-centered)
    primary_zero = pd.Series(primary_line - 50, index=df.index)
    bb_basis = primary_zero.rolling(50).mean()
    bb_std = primary_zero.rolling(50).std()
    bb_upper = bb_basis + 0.35 * bb_std
    bb_lower = bb_basis - 0.35 * bb_std
    
    secondary_zero = pd.Series(secondary_rsi - 50, index=df.index)
    primary_rsi_zero = pd.Series(primary_rsi - 50, index=df.index)
    
    threshold = 3.0
    
    # Blue signal: secondary > +threshold AND primary > bb_upper
    qqe_blue = (secondary_zero > threshold) & (primary_rsi_zero > bb_upper)
    
    # Red signal: secondary < -threshold AND primary < bb_lower  
    qqe_red = (secondary_zero < -threshold) & (primary_rsi_zero < bb_lower)
    
    return qqe_blue.values, qqe_red.values

# ═══════════ SSL HYBRID ═══════════
def compute_ssl_signals(df):
    """Returns: ssl_bull_cross, ssl_bear_cross boolean arrays"""
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    
    # Baseline: HMA(close, 60)
    baseline = hma(df['close'], 60).values
    
    # SSL Exit: HMA-based, len=15
    exit_high = hma(df['high'], 15).values
    exit_low = hma(df['low'], 15).values
    
    n = len(c)
    hlv3 = np.zeros(n)
    ssl_exit = np.full(n, np.nan)
    
    warmup = 100
    for i in range(1, n):
        if np.isnan(exit_high[i]) or np.isnan(exit_low[i]):
            hlv3[i] = hlv3[i-1] if i > 0 else 0
        elif c[i] > exit_high[i]:
            hlv3[i] = 1
        elif c[i] < exit_low[i]:
            hlv3[i] = -1
        else:
            hlv3[i] = hlv3[i-1]
        
        ssl_exit[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
    
    # Cross signals
    ssl_bull = np.zeros(n, dtype=bool)
    ssl_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if np.isnan(ssl_exit[i]) or np.isnan(ssl_exit[i-1]):
            continue
        ssl_bull[i] = c[i] > ssl_exit[i] and c[i-1] <= ssl_exit[i-1]
        ssl_bear[i] = c[i] < ssl_exit[i] and c[i-1] >= ssl_exit[i-1]
    
    return ssl_bull, ssl_bear, baseline

# ═══════════ COMBINED STRATEGY ═══════════
def backtest_combined(df):
    warmup = 200
    c = df['close'].values
    
    # EMA 200
    ema200 = ema(df['close'], 200).values
    
    # QQE signals
    qqe_blue, qqe_red = compute_qqe_signals(df)
    
    # SSL signals
    ssl_bull, ssl_bear, baseline = compute_ssl_signals(df)
    
    # Combined entry
    long_entry = np.zeros(len(c), dtype=bool)
    short_entry = np.zeros(len(c), dtype=bool)
    
    for i in range(warmup, len(c)):
        if np.isnan(ema200[i]):
            continue
        # LONG: QQE blue + SSL bull cross + close > EMA200
        if qqe_blue[i] and ssl_bull[i] and c[i] > ema200[i]:
            long_entry[i] = True
        # SHORT: QQE red + SSL bear cross + close < EMA200
        elif qqe_red[i] and ssl_bear[i] and c[i] < ema200[i]:
            short_entry[i] = True
    
    # Trade simulation: long-only with exit on opposite
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(c)):
        if pos == 0:
            if long_entry[i]:
                pos, ep = 1, c[i]
            elif short_entry[i]:
                pos, ep = -1, c[i]
        elif pos == 1:
            # Exit on short signal
            if short_entry[i]:
                pnl = (c[i]/ep-1)*100-COMM*100
                trades.append({'pnl':pnl}); eq*=(1+pnl/100)
                pos, ep = -1, c[i]
        elif pos == -1:
            if long_entry[i]:
                pnl = (1-c[i]/ep)*100-COMM*100
                trades.append({'pnl':pnl}); eq*=(1+pnl/100)
                pos, ep = 1, c[i]
        curve.append(eq)
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl}); eq*=(1+pnl/100); curve.append(eq)
    
    return trades, curve, long_entry.sum(), short_entry.sum()

def metrics(trades, curve):
    if not trades: return None
    n = len(trades)
    pnls = [t['pnl'] for t in trades]
    w = [p for p in pnls if p > 0]
    l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    eq_curve = pd.Series(curve)
    dds = ((eq_curve-eq_curve.expanding().max())/eq_curve.expanding().max()*100).min()
    dr = eq_curve.pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    fe = curve[-1]
    ann = (fe/CAP)**(365/DAYS)-1
    tp = sum(w)
    tl = sum(l)
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l),'tp':tp,'tl':tl}

# ═══════════ RUN ═══════════
TFS = ['15m', '1h', '4h']
print("Fetching FET/USDT...")
data = {}
for tf in TFS:
    data[tf] = fetch(tf)
    print(f"  {tf}: {len(data[tf])} candles")

print(f"\n{'='*80}")
print("QQE MOD + SSL Hybrid + EMA200 — FET/USDT")
print(f"{'='*80}")

for tf in TFS:
    df = data[tf]
    print(f"\n─── {tf} ───")
    trades, curve, n_long, n_short = backtest_combined(df)
    m = metrics(trades, curve)
    if m and m['n'] > 0:
        print(f"  📋 صفقات: {m['n']} | 🟢 {m['nw']} | 🔴 {m['nl']} | 📈 WR: {m['wr']:.1f}%")
        print(f"  💵 ربح: +{m['tp']:.2f}% | 💸 خسارة: {m['tl']:.2f}% | 💰 صافي: {m['tp']+m['tl']:.2f}%")
        print(f"  🟢 م.ربح: +{m['aw']:.3f}% | 🔴 م.خسارة: {m['al']:.3f}% | 📊 R:R: {m['rr']:.2f}x")
        print(f"  📊 شارپ: {m['sh']:.2f} | 📉 سحب: {m['dd']:.1f}%")
        print(f"  🏦 المحفظة: ${CAP} → ${m['eq']:.0f} (+{(m['eq']/CAP-1)*100:.1f}%) | 📈 سنوي: {m['annual']:.1f}%")
        print(f"  🔔 إشارات: 🟢شراء {n_long} | 🔴بيع {n_short}")
    else:
        print(f"  ❌ صفر صفقات")
