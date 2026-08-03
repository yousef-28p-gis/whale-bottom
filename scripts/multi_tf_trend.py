#!/usr/bin/env python3
"""
Multi-TF Trend Alignment — FET
Daily/4h/1h trend UP + 15m steep angle entry
"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

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

def slope(c, period):
    y = c[-period:]; x = np.arange(period)
    return np.polyfit(x, y, 1)[0] / np.mean(y) * 100

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

print('Fetching multi-TF data...')
d1 = fetch('1d', DAYS)     # daily
d4 = fetch('4h', DAYS)     # 4h
d1h = fetch('1h', DAYS)    # 1h  
d15 = fetch('15m', DAYS)   # 15m entry TF

print(f'  1d:{len(d1)} 4h:{len(d4)} 1h:{len(d1h)} 15m:{len(d15)}')

# Trend filters on each TF
ema50_d = ema(d1['close'], 50).values
ema50_4h = ema(d4['close'], 50).values
ema50_1h = ema(d1h['close'], 50).values

c_daily = d1['close'].values
c_4h = d4['close'].values
c_1h = d1h['close'].values

# 15m entry: steep angle + pullback
c15 = d15['close'].values; h15 = d15['high'].values; l15 = d15['low'].values
o15 = d15['open'].values; n15 = len(c15)

# Pre-compute 15m slopes
lookback = 10; slopes15 = np.full(n15, np.nan); pullbacks15 = np.full(n15, np.nan)
for i in range(50, n15):
    slopes15[i] = slope(c15[i-lookback+1:i+1], lookback)
    peak5 = h15[i-5:i+1].max()
    pullbacks15[i] = (peak5 - c15[i]) / peak5 * 100 if peak5 > 0 else 0

# Align higher TF data to 15m index
idx15 = d15.index
ema50_d_aligned = pd.Series(ema50_d, index=d1.index).reindex(idx15, method='ffill').values
ema50_4h_aligned = pd.Series(ema50_4h, index=d4.index).reindex(idx15, method='ffill').values
ema50_1h_aligned = pd.Series(ema50_1h, index=d1h.index).reindex(idx15, method='ffill').values
c_d_aligned = pd.Series(c_daily, index=d1.index).reindex(idx15, method='ffill').values
c_4h_aligned = pd.Series(c_4h, index=d4.index).reindex(idx15, method='ffill').values
c_1h_aligned = pd.Series(c_1h, index=d1h.index).reindex(idx15, method='ffill').values

# Entry on 15m: ALL higher TFs in uptrend + steep angle pattern
le = np.zeros(n15, bool)
for i in range(200, n15):
    if np.isnan(slopes15[i]) or np.isnan(ema50_d_aligned[i]): continue
    
    # Multi-TF trend alignment
    trend_1d = c_d_aligned[i] > ema50_d_aligned[i] and ema50_d_aligned[i] > ema50_d_aligned[max(0,i-96)]
    trend_4h = c_4h_aligned[i] > ema50_4h_aligned[i]
    trend_1h = c_1h_aligned[i] > ema50_1h_aligned[i]
    
    if not (trend_1d and trend_4h and trend_1h): continue
    
    # 15m steep angle entry
    steep = any(not np.isnan(slopes15[j]) and slopes15[j] > 0.3 for j in range(max(0,i-5), i+1))
    pb_ok = pullbacks15[i] > 0.5 and pullbacks15[i] < 2.5
    
    if steep and pb_ok and c15[i] > h15[i-1] and c15[i] > o15[i]:
        le[i] = True

# Simulate with TP/SL
print(f'\nSignals: {le.sum()}')
print(f'{"Config":<20} {"T":>5} {"WR":>7} {"R:R":>6} {"DD":>7} {"Profit":>9}')
print('-'*60)

for tp in [2.0, 2.5, 3.0, 4.0, 5.0]:
 for sl in [1.0, 1.5, 2.0]:
  if sl >= tp: continue
  trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
  for i in range(200, n15):
    if pos==1:
        if h15[i]>=ep*(1+tp/100):
            pnl=(tp-COMM*100); trades.append(pnl); eq*=(1+pnl/100); pos=0
        elif c15[i]<=ep*(1-sl/100):
            pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
    if pos==0 and le[i]: pos=1; ep=c15[i]
    curve.append(eq)
  if pos:
    pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
  if len(trades)<3: continue
  w2=[p for p in trades if p>0]; l2=[p for p in trades if p<=0]
  wr=len(w2)/len(trades)*100
  aw=np.mean(w2) if w2 else 0; al=np.mean(l2) if l2 else 0
  rr=abs(aw/al) if al else 99
  dd=((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
  ico='+' if eq>CAP else '-'
  print(f'TP{tp}/SL{sl:<6} {len(trades):>5} {wr:>6.1f}% {rr:>5.2f}x {dd:>6.1f}% {ico}${eq-1000:>+8.0f}')

# Compare: without multi-TF filter (just 15m pattern)
le_nofilter = np.zeros(n15, bool)
for i in range(200, n15):
    if np.isnan(slopes15[i]): continue
    steep = any(not np.isnan(slopes15[j]) and slopes15[j] > 0.3 for j in range(max(0,i-5), i+1))
    pb_ok = pullbacks15[i] > 0.5 and pullbacks15[i] < 2.5
    if steep and pb_ok and c15[i] > h15[i-1] and c15[i] > o15[i]:
        le_nofilter[i] = True

print(f'\n--- WITHOUT multi-TF filter ({le_nofilter.sum()} signals) ---')
for tp,sl in [(3.0,1.5),(4.0,2.0)]:
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200, n15):
        if pos==1:
            if h15[i]>=ep*(1+tp/100):
                pnl=(tp-COMM*100); trades.append(pnl); eq*=(1+pnl/100); pos=0
            elif c15[i]<=ep*(1-sl/100):
                pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
        if pos==0 and le_nofilter[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    w2=[p for p in trades if p>0]; l2=[p for p in trades if p<=0]
    wr=len(w2)/len(trades)*100
    ico='+' if eq>CAP else '-'
    print(f'TP{tp}/SL{sl:<6} {len(trades):>5} {wr:>6.1f}% {ico}${eq-1000:>+8.0f}')
