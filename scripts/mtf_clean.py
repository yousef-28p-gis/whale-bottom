#!/usr/bin/env python3
"""
Multi-TF Clean Approaches — No locked indicators
FET — 180 days — 15m entry
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

print('Fetching...')
d1 = fetch('1d', DAYS); d4 = fetch('4h', DAYS); d1h = fetch('1h', DAYS); d15 = fetch('15m', DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
v15=d15['volume'].values; n15=len(c15)
idx15 = d15.index

# Align higher TF to 15m
def align(higher, idx15):
    return pd.Series(higher.values, index=higher.index).reindex(idx15, method='ffill')

c1d = align(d1['close'], idx15).values; o1d = align(d1['open'], idx15).values
c4h = align(d4['close'], idx15).values; o4h = align(d4['open'], idx15).values
h4h = align(d4['high'], idx15).values; l4h = align(d4['low'], idx15).values
c1h = align(d1h['close'], idx15).values; o1h = align(d1h['open'], idx15).values
h1h = align(d1h['high'], idx15).values; l1h = align(d1h['low'], idx15).values
v1h = align(d1h['volume'], idx15).values

# EMA on 15m
ema15_20 = pd.Series(c15).ewm(span=20, adjust=False).mean().values
ema15_50 = pd.Series(c15).ewm(span=50, adjust=False).mean().values

# 15m swing lows (last 10 bars)
swing_low_15 = pd.Series(l15).rolling(10).min().values

def sim(le, tp, sl):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200, n15):
        if pos==1:
            if h15[i]>=ep*(1+tp/100):
                trades.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0
            elif c15[i]<=ep*(1-sl/100):
                pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve, eq

def met(tr, cv, eq):
    if len(tr)<3: return None
    w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    return wr,dd,eq,aw,al

# ═══════════ APPROACHES ═══════════
print(f'\n{"="*80}')
print(f'Multi-TF Clean Approaches — FET 15m')
print(f'{"="*80}')
print(f'{"Approach":<30} {"Sigs":>5} {"T":>5} {"WR":>7} {"R:R":>6} {"DD":>7} {"Profit":>9}')
print('-'*75)

# A1: Daily green + 4h green + 1h green + 15m pullback to swing low
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i] and c4h[i]>o4h[i] and c1h[i]>o1h[i]:
        if c15[i]<=swing_low_15[i]*1.01 and c15[i]>c15[i-1] and c15[i]>o15[i]:
            le[i]=True
for tp,sl in [(2.5,1.5),(3.0,2.0),(4.0,2.0)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A1 3xGreen+SwLow":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')

# A2: Daily above open + 4h above open + 1h pullback + 15m bounce off EMA20
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        # 1h pullback: current 1h low near previous 1h low
        if c15[i]<=ema15_20[i]*1.005 and c15[i]>ema15_20[i]*0.99 and c15[i]>c15[i-1]:
            le[i]=True
for tp,sl in [(2.5,1.5),(3.0,2.0)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A2 D+4hUp+1hPB+EMA20":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')

# A3: Daily above open + 4h HH/HL + 15m volume spike at support
le=np.zeros(n15,bool)
for i in range(200,n15):
    # 4h making higher highs
    hh4h = h4h[i] > h4h[max(0,i-96)]
    if c1d[i]>o1d[i] and hh4h:
        # 15m: high volume + near swing low + green candle
        if v15[i]>v15[i-10:i].mean()*1.5 and c15[i]<=swing_low_15[i]*1.01 and c15[i]>o15[i]:
            le[i]=True
for tp,sl in [(2.5,1.5),(3.0,2.0),(4.0,2.0)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A3 D+4hHH+Vol+SwLow":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')

# A4: Price above daily high + 4h pullback + 15m inside bar breakout
le=np.zeros(n15,bool)
# Pre-compute inside bars
inside15 = np.zeros(n15,bool)
for i in range(10,n15): inside15[i]=h15[i]<h15[i-1] and l15[i]>l15[i-1]
for i in range(200,n15):
    # Daily: price above yesterday's high (strong)
    if c1d[i]>h1h[max(0,i-96)]:  # today above yesterday high
        # 4h pullback: current 4h below previous 4h high
        if c4h[i]<h4h[max(0,i-96)]:
            # 15m: inside bar breakout up
            if inside15[i-1] and c15[i]>h15[i-1]:
                le[i]=True
for tp,sl in [(2.5,1.5),(3.0,2.0)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A4 AboveYH+4hPB+InsBreak":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')

# A5: Simplest — just 1d direction + 15m pullback (no 4h/1h complexity)
le=np.zeros(n15,bool)
for i in range(200,n15):
    # Daily trend: EMA20 slope up on daily
    if c1d[i]>o1d[i]:
        # 15m: low below EMA50 (oversold in uptrend) + reversal candle
        if l15[i]<=ema15_50[i]*0.995 and c15[i]>o15[i] and c15[i]>c15[i-1]:
            le[i]=True
for tp,sl in [(2.5,1.5),(3.0,2.0),(4.0,2.0),(5.0,2.5)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A5 1dUp+15mPB@EMA50":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')

# A6: 4h direction + 15m spike at support (like our best concept)
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c4h[i]>o4h[i] and c1h[i]>o1h[i]:
        # 15m spike: volume > 2x avg + at 10-bar low
        vol_ok = v15[i]>v15[i-20:i].mean()*2
        at_low = l15[i]<=l15[i-10:i].min()*1.005
        if vol_ok and at_low and c15[i]>c15[i-1]:
            le[i]=True
for tp,sl in [(1.5,1.0),(2.0,1.5),(3.0,2.0)]:
    tr,cv,eq=sim(le,tp,sl); m=met(tr,cv,eq)
    if m: print(f'{"A6 4h+1hUp+Spike+Low":<30} {le.sum():>5} {len(tr):>5} {m[0]:>6.1f}% {abs(m[3]/m[4]) if m[4] else 0:>5.2f}x {m[1]:>6.1f}% {("+" if eq>CAP else "-")}${eq-1000:>+8.0f}')
