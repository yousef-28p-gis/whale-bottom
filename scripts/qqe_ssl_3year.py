#!/usr/bin/env python3
"""
QQE+SSL+EMA — 3-Year Validation — FET/USDT
Top configs from grid search
"""
import ccxt, pandas as pd, numpy as np, sys
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
    half = int(l/2); sq = int(np.sqrt(l))
    w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1,half+1)), raw=True)
    w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1,l+1)), raw=True)
    return (2*w1 - w2).rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

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
    primary_rsi, _ = compute_qqe(c, 6, 5, 3.0)
    secondary_rsi, _ = compute_qqe(c, 6, 5, 1.61)
    
    primary_zero = pd.Series(primary_rsi - 50, index=df.index)
    bb_basis = primary_zero.rolling(50).mean()
    bb_std = primary_zero.rolling(50).std()
    bb_upper = bb_basis + 0.35 * bb_std
    bb_lower = bb_basis - 0.35 * bb_std
    
    secondary_zero = pd.Series(secondary_rsi - 50, index=df.index)
    primary_rsi_zero = pd.Series(primary_rsi - 50, index=df.index)
    qqe_blue = (secondary_zero > 3.0) & (primary_rsi_zero > bb_upper)
    qqe_red = (secondary_zero < -3.0) & (primary_rsi_zero < bb_lower)
    
    exit_high = hma(df['high'], 15).values
    exit_low = hma(df['low'], 15).values
    n = len(c); cl = c.values
    hlv3 = np.zeros(n); ssl_exit = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(exit_high[i]): hlv3[i] = hlv3[i-1]
        elif cl[i] > exit_high[i]: hlv3[i] = 1
        elif cl[i] < exit_low[i]: hlv3[i] = -1
        else: hlv3[i] = hlv3[i-1]
        ssl_exit[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
    ssl_bull = np.zeros(n, dtype=bool); ssl_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if not np.isnan(ssl_exit[i]):
            ssl_bull[i] = cl[i] > ssl_exit[i] and cl[i-1] <= ssl_exit[i-1]
            ssl_bear[i] = cl[i] < ssl_exit[i] and cl[i-1] >= ssl_exit[i-1]
    return qqe_blue.values, qqe_red.values, ssl_bull, ssl_bear

def backtest(df, ema_len, exit_mode, tp=None, sl=None, trail=None):
    warmup = 200; c = df['close'].values; h = df['high'].values; l = df['low'].values
    ema_line = ema(df['close'], ema_len).values
    qqe_blue, qqe_red, ssl_bull, ssl_bear = compute_signals(df)
    n = len(c)
    
    long_entry = np.zeros(n, dtype=bool); short_entry = np.zeros(n, dtype=bool)
    for i in range(warmup, n):
        if np.isnan(ema_line[i]): continue
        if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]: long_entry[i] = True
        elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]: short_entry[i] = True
    
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

def calc_metrics(trades, curve):
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
    tps = sum(1 for t in trades if t.get('exit')=='TP')
    sls = sum(1 for t in trades if t.get('exit')=='SL')
    trs = sum(1 for t in trades if t.get('exit')=='TRAIL')
    revs = sum(1 for t in trades if t.get('exit')=='REV')
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l),
            'tp':tps,'sl':sls,'trail':trs,'rev':revs,'net':sum(w)+sum(l)}

# ═══════════ TOP CONFIGS ═══════════
CONFIGS = [
    # (tf, ema, exit_mode, tp, sl, trail, label)
    ('4h', 50, 'tp_sl', 3.0, 1.5, None, '4h EMA50 TP3% SL1.5%'),
    ('4h', 100, 'tp_sl', 3.0, 1.5, None, '4h EMA100 TP3% SL1.5%'),
    ('4h', 50, 'reverse', None, None, None, '4h EMA50 reverse'),
    ('4h', 50, 'trail', None, None, 0.5, '4h EMA50 trail0.5%'),
    ('1h', 100, 'reverse', None, None, None, '1h EMA100 reverse'),
    ('1h', 50, 'reverse', None, None, None, '1h EMA50 reverse'),
    ('1h', 200, 'reverse', None, None, None, '1h EMA200 reverse'),
    ('1h', 50, 'tp_sl', 5.0, 2.0, None, '1h EMA50 TP5% SL2%'),
]

print("Fetching 3 years FET/USDT 1h + 4h...")
data = {}
for tf in ['1h', '4h']:
    data[tf] = fetch(tf, DAYS)
    print(f"  {tf}: {len(data[tf])} candles")

print(f"\n{'='*90}")
print(f"QQE+SSL+EMA — 3-YEAR validation — FET/USDT")
print(f"📅 2023-08-01 → 2026-08-01 | Commission 0.2%")
print(f"{'='*90}")

all_res = []
for tf, ema_len, exit_mode, tp, sl, trail, label in CONFIGS:
    df = data[tf]
    trades, curve = backtest(df, ema_len, exit_mode, tp, sl, trail)
    m = calc_metrics(trades, curve)
    if m and m['n'] > 0:
        m['label'] = label; all_res.append(m)
        e = f"🎯{m['tp']} 🛑{m['sl']} 🐌{m['trail']} 🔄{m['rev']}"
        print(f"\n─── {label} ───")
        print(f"  📋 صفقات: {m['n']} | 🟢 {m['nw']} | 🔴 {m['nl']} | 📈 WR: {m['wr']:.1f}%")
        print(f"  🟢 م.ربح: +{m['aw']:.2f}% | 🔴 م.خسارة: {m['al']:.2f}% | 📊 R:R: {m['rr']:.2f}x")
        print(f"  📊 شارپ: {m['sh']:.2f} | 📉 سحب: {m['dd']:.1f}%")
        print(f"  🏦 ${CAP} → ${m['eq']:.0f} (+{(m['eq']/CAP-1)*100:.1f}%) | 📈 سنوي: {m['annual']:.1f}%")
        print(f"  {e}")

# Yearly breakdown for best config
print(f"\n{'='*90}")
print("YEARLY BREAKDOWN — Best Config (4h EMA50 TP3% SL1.5%)")
print(f"{'='*90}")

df = data['4h']
trades, curve = backtest(df, 50, 'tp_sl', 3.0, 1.5, None)
if trades:
    # Map trades to years
    trade_years = {}
    idx = df.index
    ti = 0
    # Find which year each trade belongs to
    warmup = 200
    # Re-simulate with timestamps for yearly breakdown
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    ema_line = ema(df['close'], 50).values
    qqe_blue, qqe_red, ssl_bull, ssl_bear = compute_signals(df)
    n = len(c)
    
    yearly = {}
    pos = 0; ep = 0; peak = 0
    for i in range(warmup, n):
        year = idx[i].year
        if year not in yearly:
            yearly[year] = {'trades': [], 'eq_start': CAP if not yearly else list(yearly.values())[-1].get('eq_end', CAP)}
        
        if pos == 0:
            if not np.isnan(ema_line[i]):
                if qqe_blue[i] and ssl_bull[i] and c[i] > ema_line[i]:
                    pos=1; ep=c[i]; peak=h[i]
                elif qqe_red[i] and ssl_bear[i] and c[i] < ema_line[i]:
                    pos=-1; ep=c[i]; peak=l[i]
        elif pos == 1:
            exit_now = False; exit_px = c[i]
            if h[i] >= ep*1.03: exit_now=True; exit_px=ep*1.03
            elif c[i] <= ep*0.985: exit_now=True; exit_px=c[i]
            if exit_now:
                pnl = (exit_px/ep-1)*100-COMM*100
                yearly[year]['trades'].append(pnl)
                pos=0; peak=0
        elif pos == -1:
            exit_now = False; exit_px = c[i]
            if l[i] <= ep*0.97: exit_now=True; exit_px=ep*0.97
            elif c[i] >= ep*1.015: exit_now=True; exit_px=c[i]
            if exit_now:
                pnl = (1-exit_px/ep)*100-COMM*100
                yearly[year]['trades'].append(pnl)
                pos=0; peak=0
    
    for year in sorted(yearly.keys()):
        yr = yearly[year]
        tr = yr['trades']
        if not tr: continue
        n_tr = len(tr)
        w_tr = [t for t in tr if t > 0]
        l_tr = [t for t in tr if t <= 0]
        wr_tr = len(w_tr)/n_tr*100
        net_tr = sum(tr)
        print(f"  📅 {year}: {n_tr:>3d} صفقة | WR {wr_tr:>5.1f}% | صافي {net_tr:>+7.2f}%")

print(f"\n─── خلاصة 3 سنين ───")
print(f"أفضل تركيبة: 4h EMA50 TP3% SL1.5%")
best = all_res[0] if all_res else None
if best:
    print(f"WR: {best['wr']:.1f}% | R:R: {best['rr']:.2f}x | DD: {best['dd']:.1f}% | Sharpe: {best['sh']:.2f}")
    print(f"${CAP} → ${best['eq']:.0f} | سنوي: {best['annual']:.1f}%")
