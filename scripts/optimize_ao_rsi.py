#!/usr/bin/env python3
"""
Optimization: AO+Stoch+RSI+ATR + OCC verification
Grid search on best strategies
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

def rsi(s, p):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/p, adjust=False).mean()/l.ewm(alpha=1/p, adjust=False).mean())

def stoch(high, low, close, k_period=14, d_period=3, smooth=3):
    ll = low.rolling(k_period).min(); hh = high.rolling(k_period).max()
    k_raw = (close - ll) / (hh - ll) * 100
    return k_raw.rolling(d_period).mean(), k_raw.rolling(d_period).mean().rolling(smooth).mean()

def metrics(trades):
    if not trades: return {'n':0,'wr':0,'eq':CAP,'dd':0,'sh':0}
    n = len(trades)
    w = [t['pnl'] for t in trades if t['pnl'] > 0]
    l = [t['pnl'] for t in trades if t['pnl'] <= 0]
    wr = len(w)/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    eq_curve = [CAP]; eq = CAP
    for t in trades: eq *= (1+t['pnl']/100); eq_curve.append(eq)
    fe = eq_curve[-1]
    dds = ((pd.Series(eq_curve) - pd.Series(eq_curve).expanding().max())/pd.Series(eq_curve).expanding().max()*100).min()
    dr = pd.Series(eq_curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    ann = (fe/CAP)**(365/DAYS) - 1
    return {'n':n,'wr':wr,'eq':fe,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':len(w),'nl':len(l)}

print("Fetching 15m data...")
df = fetch('15m')
print(f"  {len(df)} candles")

hl2 = (df['high']+df['low'])/2
ao = (hl2.rolling(5).mean() - hl2.rolling(34).mean()) * 1000
k_val, d_val = stoch(df['high'], df['low'], df['close'], 14, 3, 3)
rsi_val = rsi(df['close'], 10)
atr_val = (df['high'] - df['low']).rolling(14).mean()
ao_rising = ao > ao.shift(1)

# ═══════════ GRID SEARCH: AO+Stoch+RSI+ATR ═══════════
print(f"\n{'='*90}")
print("GRID SEARCH: AO+Stoch+RSI+ATR — FET/USDT 15m — 180 days")
print(f"{'='*90}")

k_levels = [15, 20, 25]
rsi_levels = [25, 30, 35]
tp_mults = [1.0, 1.5, 2.0, 3.0]
sl_mults = [0.5, 1.0, 1.5]

c = df['close'].values; h = df['high'].values; l = df['low'].values
warmup = 200

results = []
for k_thresh, rsi_thresh, tp_m, sl_m in itertools.product(k_levels, rsi_levels, tp_mults, sl_mults):
    longs = ((k_val < k_thresh) & (rsi_val < rsi_thresh) & ao_rising).values
    shorts = ((k_val > (100-k_thresh)) & (rsi_val > (100-rsi_thresh)) & (~ao_rising)).values
    
    tp_pct = (atr_val / df['close'] * 100 * tp_m).values
    sl_pct = (atr_val / df['close'] * 100 * sl_m).values
    
    trades = []; eq = CAP; pos = 0; ep = 0
    for i in range(warmup, len(c)):
        if pos == 0:
            if longs[i]: pos, ep = 1, c[i]
            elif shorts[i]: pos, ep = -1, c[i]
        elif pos == 1:
            tpi = ep*(1+tp_pct[i]/100); sli = ep*(1-sl_pct[i]/100)
            if h[i] >= tpi:
                pnl = (tpi/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); pos=0
            elif c[i] <= sli:
                pnl = (c[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); pos=0
            elif shorts[i]:
                pnl = (c[i]/ep-1)*100-COMM*100; trades.append({'pnl':pnl}); pos,ep=-1,c[i]
        elif pos == -1:
            tpi = ep*(1-tp_pct[i]/100); sli = ep*(1+sl_pct[i]/100)
            if l[i] <= tpi:
                pnl = (1-tpi/ep)*100-COMM*100; trades.append({'pnl':pnl}); pos=0
            elif c[i] >= sli:
                pnl = (1-c[i]/ep)*100-COMM*100; trades.append({'pnl':pnl}); pos=0
            elif longs[i]:
                pnl = (1-c[i]/ep)*100-COMM*100; trades.append({'pnl':pnl}); pos,ep=1,c[i]
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append({'pnl':pnl})
    
    m = metrics(trades)
    m['config'] = f"K<{k_thresh} RSI<{rsi_thresh} TPx{tp_m} SLx{sl_m}"
    results.append(m)

# Top by WR
print("\n🏆 TOP 15 by Win Rate:")
by_wr = sorted(results, key=lambda x: x['wr'], reverse=True)
for i, r in enumerate(by_wr[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon} {r['config']:>35} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | AW {r['aw']:>+.2f}% | AL {r['al']:>+.2f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f}")

print("\n🏆 TOP 15 by Return:")
by_eq = sorted(results, key=lambda x: x['eq'], reverse=True)
for i, r in enumerate(by_eq[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon} {r['config']:>35} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f}")

print("\n🏆 TOP 15 by Sharpe:")
by_sh = sorted(results, key=lambda x: x['sh'], reverse=True)
for i, r in enumerate(by_sh[:15]):
    icon = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
    print(f"{icon} {r['config']:>35} | {r['n']:>3d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | Sharpe {r['sh']:>5.2f} | ${r['eq']-1000:>+8.0f}")

# ═══════════ OCC FIXED-SIZE CHECK ═══════════
print(f"\n{'='*90}")
print("OCC Strategy — Fixed $500/trade (NO compounding)")
print(f"{'='*90}")

htf = df.resample('3h').agg({'open':'first','close':'last'}).dropna()
htf_aligned = htf.reindex(df.index, method='ffill')
htf_cross_up = (htf_aligned['close'] > htf_aligned['open']) & (htf_aligned['close'].shift(1) <= htf_aligned['open'].shift(1))
htf_cross_down = (htf_aligned['close'] < htf_aligned['open']) & (htf_aligned['close'].shift(1) >= htf_aligned['open'].shift(1))

longs = htf_cross_up.values; shorts = htf_cross_down.values
trades_fixed = []; pos = 0; ep = 0; TRADE_SIZE = 500
for i in range(200, len(c)):
    if pos == 0:
        if longs[i]: pos, ep = 1, c[i]
        elif shorts[i]: pos, ep = -1, c[i]
    elif pos == 1:
        if shorts[i]:
            pnl_pct = (c[i]/ep-1)*100-COMM*100
            pnl_usd = TRADE_SIZE * pnl_pct/100
            trades_fixed.append({'pnl':pnl_pct})
            pos, ep = -1, c[i]
    elif pos == -1:
        if longs[i]:
            pnl_pct = (1-c[i]/ep)*100-COMM*100
            pnl_usd = TRADE_SIZE * pnl_pct/100
            trades_fixed.append({'pnl':pnl_pct})
            pos, ep = 1, c[i]

total_pnl_usd = sum(t['pnl'] for t in trades_fixed)
m_fixed = metrics(trades_fixed)
print(f"  Trades: {m_fixed['n']} | WR: {m_fixed['wr']:.1f}% | AW: {m_fixed['aw']:+.2f}% | AL: {m_fixed['al']:+.2f}%")
print(f"  Total PnL (no compounding): ${total_pnl_usd:+.0f} on ${TRADE_SIZE}/trade")
print(f"  R:R: {m_fixed['rr']:.2f}x | DD: {m_fixed['dd']:.1f}% | Sharpe: {m_fixed['sh']:.2f}")

# Check a few trades for validity
print(f"\n  Sample trades:")
samples = trades_fixed[:5] + trades_fixed[-5:] if len(trades_fixed) >= 10 else trades_fixed
for i, t in enumerate(samples):
    print(f"    Trade {i+1}: PnL {t['pnl']:+.2f}%")
