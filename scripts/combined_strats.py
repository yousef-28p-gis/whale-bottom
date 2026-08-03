#!/usr/bin/env python3
"""
Combined Strategies: QQE+SSL+EMA + filters from today's strategies
FET/USDT 1h — 180 days
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

def stoch(high, low, close, kp=14, dp=3, sp=3):
    ll = low.rolling(kp).min(); hh = high.rolling(kp).max()
    k_raw = (close - ll) / (hh - ll) * 100
    return k_raw.rolling(dp).mean(), k_raw.rolling(dp).mean().rolling(sp).mean()

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
    if not trades or len(trades) < 3: return None
    pnls = trades; n = len(pnls)
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    return {'n':n,'wr':wr,'eq':curve[-1],'dd':dds,'sh':sh,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

# ═══════════ FETCH ═══════════
print("Fetching FET/USDT 1h...")
df = fetch('1h', DAYS)
print(f"  {len(df)} candles")
c = df['close'].values; h = df['high'].values; l = df['low'].values
n = len(c); warmup = 200

# ═══════════ BASE: QQE+SSL+EMA ═══════════
print("Computing QQE+SSL+EMA200...")
primary_rsi = compute_qqe(df['close'], 6, 5, 2.0)
secondary_rsi = compute_qqe(df['close'], 6, 5, 1.61)
primary_zero = primary_rsi - 50; secondary_zero = secondary_rsi - 50

bb_basis = pd.Series(primary_zero).rolling(30).mean().values
bb_std = pd.Series(primary_zero).rolling(30).std().values
bb_upper = bb_basis + 0.5 * bb_std; bb_lower = bb_basis - 0.5 * bb_std

qqe_blue = (secondary_zero > 2.0) & (primary_zero > bb_upper)
qqe_red = (secondary_zero < -2.0) & (primary_zero < bb_lower)

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

ema200 = ema(df['close'], 200).values
ema50 = ema(df['close'], 50).values

# ═══════════ FILTERS from other strategies ═══════════
print("Computing additional filters...")

# Filter 1: BB+RSI — BB(200,2) + RSI(6) crossover 50
bb200_basis = df['close'].rolling(200).mean().values
bb200_std = df['close'].rolling(200).std().values
bb200_upper = bb200_basis + 2*bb200_std
bb200_lower = bb200_basis - 2*bb200_std
rsi6 = rsi_s(df['close'], 6).values
f1_bull = (rsi6 > 50) & (rsi6_prev := np.roll(rsi6, 1)) & (rsi6_prev <= 50)if False else (rsi6 > 50) & (np.roll(rsi6, 1) <= 50)
f1_bear = (rsi6 < 50) & (np.roll(rsi6, 1) >= 50)
# BB touch confirmation
bb_touch_upper = (c > bb200_upper) & (np.roll(c, 1) <= bb200_upper)
bb_touch_lower = (c < bb200_lower) & (np.roll(c, 1) >= bb200_lower)
f1_long = np.zeros(n, dtype=bool); f1_short = np.zeros(n, dtype=bool)
for i in range(warmup, n):
    if np.isnan(bb200_upper[i]): continue
    # RSI cross 50 + price at BB extreme
    if (rsi6[i] > 50 and rsi6[i-1] <= 50) and (c[i-1] < bb200_lower[i-1] and c[i] > bb200_lower[i-1]):
        f1_long[i] = True
    if (rsi6[i] < 50 and rsi6[i-1] >= 50) and (c[i-1] > bb200_upper[i-1] and c[i] < bb200_upper[i-1]):
        f1_short[i] = True

# Filter 2: AO+Stoch+RSI
hl2 = (df['high'] + df['low']) / 2
ao = (hl2.rolling(5).mean() - hl2.rolling(34).mean()).values * 1000
k_val, _ = stoch(df['high'], df['low'], df['close'], 14, 3, 3)
k_arr = k_val.values; rsi10 = rsi_s(df['close'], 10).values
f2_long = np.zeros(n, dtype=bool); f2_short = np.zeros(n, dtype=bool)
for i in range(warmup, n):
    if np.isnan(ao[i]): continue
    # K<20, RSI<30, AO rising
    if k_arr[i] < 20 and rsi10[i] < 30 and ao[i] > ao[i-1]:
        f2_long[i] = True
    # K>80, RSI>70, AO falling
    if k_arr[i] > 80 and rsi10[i] > 70 and ao[i] < ao[i-1]:
        f2_short[i] = True

# Filter 3: MACD+SMA200
fast_ma = df['close'].rolling(12).mean().values
slow_ma = df['close'].rolling(26).mean().values
macd = fast_ma - slow_ma; sig = pd.Series(macd).rolling(9).mean().values
hist = macd - sig; sma200 = df['close'].rolling(200).mean().values
f3_long = np.zeros(n, dtype=bool); f3_short = np.zeros(n, dtype=bool)
for i in range(warmup, n):
    if np.isnan(hist[i]): continue
    if hist[i] > 0 and hist[i-1] <= 0 and macd[i] > 0 and fast_ma[i] > slow_ma[i] and c[max(0,i-26)] > sma200[i]:
        f3_long[i] = True
    if hist[i] < 0 and hist[i-1] >= 0 and macd[i] < 0 and fast_ma[i] < slow_ma[i] and c[max(0,i-26)] < sma200[i]:
        f3_short[i] = True

# Filter 4: Flawless Victory — BB(20,1.0) touch + RSI(14)>42
bb20_basis = df['close'].rolling(20).mean().values
bb20_std = df['close'].rolling(20).std().values
bb20_upper = bb20_basis + 1.0*bb20_std; bb20_lower = bb20_basis - 1.0*bb20_std
rsi14 = rsi_s(df['close'], 14).values
f4_long = np.zeros(n, dtype=bool); f4_short = np.zeros(n, dtype=bool)
for i in range(warmup, n):
    if np.isnan(bb20_lower[i]): continue
    if c[i] < bb20_lower[i] and rsi14[i] > 42:
        f4_long[i] = True
    if c[i] > bb20_upper[i] and rsi14[i] > 70:
        f4_short[i] = True

# ═══════════ COMBINATIONS ═══════════
print("\nTesting combinations...\n")

def build_entries(base_blue, base_red, ssl_b, ssl_be, ema_l, extra_long=None, extra_short=None, need_ema=True):
    le = np.zeros(n, dtype=bool); se = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if need_ema and np.isnan(ema_l[i]): continue
        base_l = base_blue[i] and ssl_b[i] and (not need_ema or c[i] > ema_l[i])
        base_s = base_red[i] and ssl_be[i] and (not need_ema or c[i] < ema_l[i])
        if extra_long is not None:
            base_l = base_l and extra_long[i]
            base_s = base_s and extra_short[i]
        if base_l: le[i] = True
        elif base_s: se[i] = True
    return le, se

combos = [
    ("1-Base QQE+SSL+EMA200 REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, None, None, True, 'reverse',None,None),
    ("2-Base + BB-RSI filter REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f1_long, f1_short, True, 'reverse',None,None),
    ("3-Base + AO-Stoch-RSI REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f2_long, f2_short, True, 'reverse',None,None),
    ("4-Base + MACD-SMA200 REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f3_long, f3_short, True, 'reverse',None,None),
    ("5-Base + Flawless REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f4_long, f4_short, True, 'reverse',None,None),
    ("6-Base + BB-RSI TP3/SL2", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f1_long, f1_short, True, 'tp_sl',3.0,2.0),
    ("7-Base + AO TP3/SL2", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, f2_long, f2_short, True, 'tp_sl',3.0,2.0),
    ("8-Base + BB-RSI+MACD REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, (f1_long & f3_long), (f1_short & f3_short), True, 'reverse',None,None),
    ("9-Base + ALL filters REV", qqe_blue, qqe_red, ssl_bull, ssl_bear, ema200, (f1_long & f2_long & f3_long & f4_long), (f1_short & f2_short & f3_short & f4_short), True, 'reverse',None,None),
]

for name, qb, qr, sb, sbe, ema_l, fl, fs, need_ema, exit_mode, tp, sl in combos:
    le, se = build_entries(qb, qr, sb, sbe, ema_l, fl, fs, need_ema)
    sigs = le.sum() + se.sum()
    trades, curve = simulate(c, h, l, le, se, exit_mode, tp, sl)
    m = metrics(trades, curve)
    if m:
        icon = "✅" if m['eq'] > 1000 else "🔴"
        print(f"{icon} {name:<35} | {m['n']:>3d}t ({sigs} signals) | WR {m['wr']:>5.1f}% | R:R {m['rr']:.2f}x | DD {m['dd']:>6.1f}% | ${m['eq']-1000:>+8.0f}")
    else:
        print(f"⚪ {name:<35} | 0 trades ({sigs} signals)")

# Best: detail
print(f"\n{'='*80}")
print("BEST CONFIG DETAIL:")
print(f"{'='*80}")
