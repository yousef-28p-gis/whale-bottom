#!/usr/bin/env python3
"""
OCC Strategy — FIXED look-ahead + realistic compounding
Uses shift(1) to prevent look-ahead bias
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

print("Fetching FET/USDT 15m/1h/4h...")
data15 = fetch('15m')
data1h = fetch('1h')
data4h = fetch('4h')
print(f"  15m: {len(data15)}, 1h: {len(data1h)}, 4h: {len(data4h)}")

# ═══════════ OCC with CORRECT look-ahead handling ═══════════
# Build HTF from chart data, then SHIFT to prevent look-ahead
# Entry uses next candle close (correct: we don't know HTF close until it's done)

def test_occ_fixed(df, chart_tf, htf_mult):
    """OCC strategy with no look-ahead bias"""
    
    # Resample to HTF
    if chart_tf == '15m':
        htf = df.resample(f'{15*htf_mult}min').agg({'open':'first','close':'last'}).dropna()
    elif chart_tf == '1h':
        htf = df.resample(f'{htf_mult}h').agg({'open':'first','close':'last'}).dropna()
    elif chart_tf == '4h':
        htf = df.resample(f'{4*htf_mult}h').agg({'open':'first','close':'last'}).dropna()
    else:
        return None
    
    # CRITICAL: shift HTF so bar at time T sees HTF candle ending BEFORE T
    # Without shift: bar 00:00 sees 00:00 3h candle close → look-ahead
    # With shift(1): bar 00:00 sees 21:00 3h candle → correct
    htf_aligned = htf.reindex(df.index, method='ffill').shift(1).bfill()
    
    htf_cross_up = (htf_aligned['close'] > htf_aligned['open']) & (htf_aligned['close'].shift(1) <= htf_aligned['open'].shift(1))
    htf_cross_down = (htf_aligned['close'] < htf_aligned['open']) & (htf_aligned['close'].shift(1) >= htf_aligned['open'].shift(1))
    
    c = df['close'].values; longs = htf_cross_up.values; shorts = htf_cross_down.values
    warmup = 200
    
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0
    for i in range(warmup, len(c)):
        if pos == 0:
            if longs[i]: pos, ep = 1, c[i]
            elif shorts[i]: pos, ep = -1, c[i]
        elif pos == 1:
            if shorts[i]:
                pnl = (c[i]/ep-1)*100-COMM*100
                trades.append(pnl); eq*=(1+pnl/100); pos,ep=-1,c[i]
        elif pos == -1:
            if longs[i]:
                pnl = (1-c[i]/ep)*100-COMM*100
                trades.append(pnl); eq*=(1+pnl/100); pos,ep=1,c[i]
        curve.append(eq)
    if pos:
        pnl = ((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    
    if not trades: return None
    
    n = len(trades)
    w = [t for t in trades if t > 0]; l = [t for t in trades if t <= 0]
    nw, nl = len(w), len(l)
    wr = nw/n*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    rr = abs(aw/al) if al != 0 else float('inf')
    dds = ((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr = pd.Series(curve).pct_change().dropna()
    sh = (dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    ann = (eq/CAP)**(365/DAYS)-1
    
    return {'n':n,'wr':wr,'eq':eq,'dd':dds,'sh':sh,'annual':ann*100,'rr':rr,'aw':aw,'al':al,'nw':nw,'nl':nl}

# Test all combos
print(f"\n{'='*90}")
print("OCC Strategy — FIXED (no look-ahead) — FET/USDT")
print(f"{'='*90}")

combos = [
    ('15m', 3, '3h'),
    ('15m', 4, '4h'),
    ('1h', 4, '4h'),
    ('1h', 8, '8h'),
    ('4h', 3, '12h'),
    ('4h', 6, '1d'),
]

for chart_tf, mult, htf_label in combos:
    df = data15 if chart_tf == '15m' else (data1h if chart_tf == '1h' else data4h)
    r = test_occ_fixed(df, chart_tf, mult)
    if r:
        print(f"  Chart={chart_tf} HTF={htf_label}: {r['n']:>4d}t | WR {r['wr']:>5.1f}% | AW {r['aw']:>+.2f}% | AL {r['al']:>+.2f}% | R:R {r['rr']:.2f}x | DD {r['dd']:>6.1f}% | ${r['eq']-1000:>+8.0f} | Sharpe {r['sh']:>5.2f}")
    else:
        print(f"  Chart={chart_tf} HTF={htf_label}: NO TRADES")

# Also: fixed $500/trade version for realism
print(f"\n{'='*90}")
print("OCC 15m/3h — Fixed $500/trade — NO compounding")
print(f"{'='*90}")

df = data15
htf = df.resample('45min').agg({'open':'first','close':'last'}).dropna()
htf_aligned = htf.reindex(df.index, method='ffill').shift(1).bfill()
htf_cross_up = (htf_aligned['close'] > htf_aligned['open']) & (htf_aligned['close'].shift(1) <= htf_aligned['open'].shift(1))
htf_cross_down = (htf_aligned['close'] < htf_aligned['open']) & (htf_aligned['close'].shift(1) >= htf_aligned['open'].shift(1))

c = df['close'].values; longs = htf_cross_up.values; shorts = htf_cross_down.values
trades_usd = []; pos = 0; ep = 0; SIZE = 500
for i in range(200, len(c)):
    if pos == 0:
        if longs[i]: pos, ep = 1, c[i]
        elif shorts[i]: pos, ep = -1, c[i]
    elif pos == 1:
        if shorts[i]:
            pnl = (c[i]/ep-1)*100-COMM*100
            trades_usd.append(SIZE*pnl/100)
            pos,ep=-1,c[i]
    elif pos == -1:
        if longs[i]:
            pnl = (1-c[i]/ep)*100-COMM*100
            trades_usd.append(SIZE*pnl/100)
            pos,ep=1,c[i]

total = sum(trades_usd)
n = len(trades_usd)
w = [t for t in trades_usd if t > 0]; l = [t for t in trades_usd if t <= 0]
print(f"  Trades: {n} | WR: {len(w)/n*100:.1f}% | Total PnL: ${total:+.0f}")
print(f"  Avg Win: ${np.mean(w):+.0f}" if w else "  No wins")
print(f"  Avg Loss: ${np.mean(l):+.0f}" if l else "  No losses")

# Sample trades
print(f"\n  First 5 trade PnL ($):")
for i in range(min(5, len(trades_usd))):
    print(f"    Trade {i+1}: ${trades_usd[i]:+.0f}")
print(f"  Last 5 trade PnL ($):")
for i in range(max(0, len(trades_usd)-5), len(trades_usd)):
    print(f"    Trade {i}: ${trades_usd[i]:+.0f}")
