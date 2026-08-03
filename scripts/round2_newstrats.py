#!/usr/bin/env python3
"""
Round 2: NEW Strategies Test + Optimization
6 new strategies on FET/USDT 15m/1h/4h
"""
import ccxt, pandas as pd, numpy as np, sys
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000
TFS = ['15m', '1h', '4h']

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

def rsi(s, p):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/p, adjust=False).mean()/l.ewm(alpha=1/p, adjust=False).mean())

def stoch(high, low, close, k_period=14, d_period=3, smooth=3):
    ll = low.rolling(k_period).min(); hh = high.rolling(k_period).max()
    k_raw = (close - ll) / (hh - ll) * 100
    k = k_raw.rolling(d_period).mean()
    d = k.rolling(smooth).mean()
    return k, d

def metrics(trades, eq_curve):
    if not trades: return {'n':0,'wr':0,'eq':CAP,'dd':0,'sharpe':0,'annual':0,'rr':0,'avg_w':0,'avg_l':0}
    n = len(trades)
    w = [t['pnl'] for t in trades if t['pnl'] > 0]
    l = [t['pnl'] for t in trades if t['pnl'] <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    fe = eq_curve[-1]
    eq = pd.Series(eq_curve)
    dd = ((eq - eq.expanding().max())/eq.expanding().max()*100).min()
    dr = eq.pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    ann = (fe/CAP)**(365/DAYS) - 1
    return {'n':n,'wr':wr,'eq':fe,'dd':dd,'sharpe':sh,'annual':ann*100,'rr':rr,'avg_w':aw,'avg_l':al,'nw':len(w),'nl':len(l)}

def simulate_reversal(closes, longs, shorts, warmup):
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(closes)):
        if pos == 0:
            if longs[i]: pos, ep = 1, closes[i]
            elif shorts[i]: pos, ep = -1, closes[i]
        elif pos == 1:
            if shorts[i]:
                pnl = (closes[i]/ep-1)*100 - COMM*100; trades.append({'pnl':pnl})
                eq *= (1+pnl/100); pos, ep = -1, closes[i]
        elif pos == -1:
            if longs[i]:
                pnl = (1 - closes[i]/ep)*100 - COMM*100; trades.append({'pnl':pnl})
                eq *= (1+pnl/100); pos, ep = 1, closes[i]
        curve.append(eq)
    if pos != 0:
        pnl = ((closes[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-closes[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl}); eq *= (1+pnl/100); curve.append(eq)
    return trades, curve

def simulate_tpsl(closes, highs, lows, longs, shorts, warmup, tp_pcts, sl_pcts):
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(closes)):
        if pos == 0:
            if longs[i]: pos, ep = 1, closes[i]
            elif shorts[i]: pos, ep = -1, closes[i]
        elif pos == 1:
            tp = ep*(1+tp_pcts[i]/100); sl = ep*(1-sl_pcts[i]/100)
            if highs[i] >= tp:
                pnl = (tp/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
            elif closes[i] <= sl:
                pnl = (closes[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
            elif shorts[i]:
                pnl = (closes[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos,ep=-1,closes[i]
        elif pos == -1:
            tp = ep*(1-tp_pcts[i]/100); sl = ep*(1+sl_pcts[i]/100)
            if lows[i] <= tp:
                pnl = (1-tp/ep)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
            elif closes[i] >= sl:
                pnl = (1-closes[i]/ep)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
            elif longs[i]:
                pnl = (1-closes[i]/ep)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos,ep=1,closes[i]
        curve.append(eq)
    if pos != 0:
        pnl = ((closes[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-closes[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl}); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

# ═══════════ FETCH DATA ═══════════
print("Fetching FET/USDT 15m/1h/4h...")
data = {}
for tf in TFS:
    data[tf] = fetch(tf)
    print(f"  {tf}: {len(data[tf])} candles")

# ═══════════ STRATEGIES ═══════════

def strat6_hull_suite(df):
    """Hull Suite: HMA crossover HMA[2]"""
    warmup = 200
    def hma(s, l):
        half = int(l/2); sq = int(np.sqrt(l))
        w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1, half+1)))
        w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1, l+1)))
        return (2*w1 - w2).rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1, sq+1)))
    hull = hma(df['close'], 55)
    longs = (hull > hull.shift(2)).values
    shorts = (hull < hull.shift(2)).values
    c = df['close'].values
    return simulate_reversal(c, longs, shorts, warmup)

def strat7_ao_stoch_rsi_atr(df):
    """AO+Stoch+RSI+ATR: K<20, RSI<30, AO rising => Long + ATR TP/SL"""
    warmup = 200
    hl2 = (df['high']+df['low'])/2
    ao = (hl2.rolling(5).mean() - hl2.rolling(34).mean()) * 1000  # Awesome Osc
    k, d = stoch(df['high'], df['low'], df['close'], 14, 3, 3)
    rsi_val = rsi(df['close'], 10)
    atr_val = (df['high'] - df['low']).rolling(14).mean()
    
    ao_rising = ao > ao.shift(1)
    longs = ((k < 20) & (rsi_val < 30) & ao_rising).values
    shorts = ((k > 80) & (rsi_val > 70) & (~ao_rising)).values
    
    # TP/SL = ±ATR from close
    tp_pct = (atr_val / df['close'] * 100).values
    sl_pct = tp_pct.copy()  # symmetric TP/SL
    return simulate_tpsl(df['close'].values, df['high'].values, df['low'].values, longs, shorts, warmup, tp_pct, sl_pct)

def strat8_golden_cross(df):
    """Golden Cross: SMA50 cross SMA200, long-only"""
    warmup = 200
    sma50 = df['close'].rolling(50).mean()
    sma200 = df['close'].rolling(200).mean()
    longs = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).values
    shorts = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).values
    c = df['close'].values
    # long-only with exit
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(c)):
        if pos == 0 and longs[i]: pos, ep = 1, c[i]
        elif pos == 1 and shorts[i]:
            pnl = (c[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
        curve.append(eq)
    if pos: pnl = (c[-1]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def strat9_ema_cross(df):
    """EMA Cross: EMA10 cross EMA20"""
    warmup = 100
    e10 = df['close'].ewm(span=10, adjust=False).mean()
    e20 = df['close'].ewm(span=20, adjust=False).mean()
    longs = ((e10 > e20) & (e10.shift(1) <= e20.shift(1))).values
    shorts = ((e10 < e20) & (e10.shift(1) >= e20.shift(1))).values
    return simulate_reversal(df['close'].values, longs, shorts, warmup)

def strat10_flawless_v1(df):
    """Flawless Victory v1: BB(20,1.0) touch + RSI>42 => long, close when RSI>70"""
    warmup = 200
    bb_basis = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_basis + 1.0*bb_std
    bb_lower = bb_basis - 1.0*bb_std
    rsi_val = rsi(df['close'], 14)
    longs = ((df['close'] < bb_lower) & (rsi_val > 42)).values
    shorts = ((df['close'] > bb_upper) & (rsi_val > 70)).values  # exit only
    c = df['close'].values
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(c)):
        if pos == 0 and longs[i]: pos, ep = 1, c[i]
        elif pos == 1 and shorts[i]:
            pnl = (c[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); pos=0
        curve.append(eq)
    if pos: pnl = (c[-1]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def strat11_occ(df):
    """Open Close Cross: HTF open/close crossover (3x multiplier)"""
    warmup = 200
    # Resample to 3x timeframe: 15m→45m, 1h→3h, 4h→12h
    tf_map = {'15m': '45min', '1h': '3h', '4h': '12h'}
    resample_map = {'15m': '45min', '1h': '3h', '4h': '4h'}  # fallback
    try:
        htf = df.resample('3h' if len(df) > 4000 else '12h').agg({'open':'first','close':'last'}).dropna()
    except:
        htf = df.resample('4h').agg({'open':'first','close':'last'}).dropna()
    
    # Align HTF to chart TF
    htf_aligned = htf.reindex(df.index, method='ffill')
    htf_cross_up = (htf_aligned['close'] > htf_aligned['open']) & (htf_aligned['close'].shift(1) <= htf_aligned['open'].shift(1))
    htf_cross_down = (htf_aligned['close'] < htf_aligned['open']) & (htf_aligned['close'].shift(1) >= htf_aligned['open'].shift(1))
    
    longs = htf_cross_up.values
    shorts = htf_cross_down.values
    return simulate_reversal(df['close'].values, longs, shorts, warmup)

# ═══════════ RUN ═══════════
STRATS = [
    ("6-Hull Suite", strat6_hull_suite),
    ("7-AO+Stoch+RSI+ATR", strat7_ao_stoch_rsi_atr),
    ("8-Golden Cross", strat8_golden_cross),
    ("9-EMA Cross", strat9_ema_cross),
    ("10-FlawlessV1", strat10_flawless_v1),
    ("11-OCC", strat11_occ),
]

all_res = []
print(f"\n{'='*85}")
print(f"ROUND 2 — 6 NEW Strategies — FET/USDT — {DAYS} days")
print(f"{'='*85}")

for sname, sfn in STRATS:
    print(f"\n─── {sname} ───")
    for tf in TFS:
        df = data[tf]
        trades, eq = sfn(df)
        m = metrics(trades, eq)
        m['strat'], m['tf'] = sname, tf
        all_res.append(m)
        print(f"  {tf:>4}: {m['n']:>4d}t | WR {m['wr']:>5.1f}% | R:R {m['rr']:>5.2f}x | DD {m['dd']:>6.1f}% | ${m['eq']-1000:>+8.0f} | Sharpe {m['sharpe']:>5.2f}")

# RANK
print(f"\n{'='*85}")
print("🏆 TOP 15 — by Return")
print(f"{'='*85}")
for i, r in enumerate(sorted(all_res, key=lambda x: x['eq'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['strat']:>20} {r['tf']:>4} | {r['n']:>4d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f} | {r['annual']:>+6.1f}% yr")

print(f"\n{'='*85}")
print("🏆 TOP 15 — by Win Rate")
print(f"{'='*85}")
for i, r in enumerate(sorted(all_res, key=lambda x: x['wr'], reverse=True)[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon:>3} {r['strat']:>20} {r['tf']:>4} | {r['n']:>4d}t | WR {r['wr']:>5.1f}% | R:R {r['rr']:>5.2f}x | ${r['eq']-1000:>+8.0f} | Sharpe {r['sharpe']:>5.2f}")
