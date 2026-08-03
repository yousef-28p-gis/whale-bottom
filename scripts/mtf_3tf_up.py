#!/usr/bin/env python3
"""Multi-TF — 1d+4h+1h ALL UP — Better 15m entries — 3 years"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

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

def ema(s,p): return s.ewm(span=p,adjust=False).mean()

print('Fetching...')
d1=fetch('1d',DAYS); d4=fetch('4h',DAYS); d1h=fetch('1h',DAYS); d15=fetch('15m',DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
v15=d15['volume'].values; n15=len(c15); idx15=d15.index

def a(s, idx15):
    return pd.Series(s.shift(1).values, index=s.index).reindex(idx15, method='ffill').values

c1d=a(d1['close'],idx15); o1d=a(d1['open'],idx15); h1d=a(d1['high'],idx15)
c4h=a(d4['close'],idx15); o4h=a(d4['open'],idx15)
c1h=a(d1h['close'],idx15); o1h=a(d1h['open'],idx15); l1h=a(d1h['low'],idx15)

ema50_d=a(ema(d1['close'],50),idx15); ema200_d=a(ema(d1['close'],200),idx15)

# 15m indicators
ema50_15=pd.Series(c15).ewm(span=50,adjust=False).mean().values
ema200_15=pd.Series(c15).ewm(span=200,adjust=False).mean().values
swing_low_15=pd.Series(l15).rolling(20).min().values
swing_high_15=pd.Series(h15).rolling(20).max().values
vsma20=pd.Series(v15).ewm(span=20,adjust=False).mean().values

# === TREND FILTER: 1d+4h+1h ALL bullish ===
# Strong: EMA50>EMA200 on daily + price>EMA50 on daily/4h/1h
trend_mask=np.zeros(n15,bool)
for i in range(200,n15):
    if np.isnan(ema50_d[i]): continue
    ok = ema50_d[i] > ema200_d[i] and c1d[i] > ema50_d[i]
    ok = ok and c4h[i] > o4h[i]  # 4h green
    ok = ok and c1h[i] > o1h[i]  # 1h green
    trend_mask[i] = ok

print(f'Trend days: {trend_mask.sum()/len(trend_mask)*100:.1f}%')

def sim(le, tp, sl):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200,n15):
        if pos==1:
            if h15[i]>=ep*(1+tp/100):
                trades.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0
            elif c15[i]<=ep*(1-sl/100):
                trades.append((c15[i]/ep-1)*100-COMM*100); eq*=(1+((c15[i]/ep-1)*100-COMM*100)/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve, eq

def rep(name, le, tp, sl):
    tr,cv,eq=sim(le,tp,sl)
    if len(tr)<5: return
    w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='+' if eq>CAP else '-'
    print(f'{name:<45} {le.sum():>5}s {len(tr):>4d}t WR{wr:>5.1f}% R:R{abs(aw/al) if al else 99:>5.2f}x DD{dd:>5.1f}% {ico}${eq-1000:>+8.0f}')

print(f'\n{"="*90}')
print(f'1d+4h+1h ALL UP — Different 15m entries — 3 years')
print(f'{"="*90}')

# E0: EMA20 bounce (baseline)
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    if c15[i]<=ema50_15[i]*1.005 and c15[i]>ema50_15[i]*0.99 and c15[i]>o15[i]:
        le[i]=True
rep('E0 EMA20 bounce', le, 4.0, 2.0)

# E1: Deep pullback to EMA50 + reversal
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    if l15[i]<=ema50_15[i]*0.995 and c15[i]>o15[i] and c15[i]>c15[i-1]:
        le[i]=True
rep('E1 Deep PB to EMA50 + rev candle', le, 4.0, 2.0)

# E2: Price near swing low + bullish engulfing
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    near_low = c15[i] <= swing_low_15[i]*1.005
    engulf = c15[i]>c15[i-1] and o15[i]<c15[i-1] and c15[i]>o15[i-1]
    if near_low and engulf:
        le[i]=True
rep('E2 SwingLow + engulfing', le, 4.0, 2.0)

# E3: Break above pullback high (confirmation)
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    # Price pulled back (made lower low) then breaks above recent high
    pullback = l15[i] < l15[max(0,i-5):i].min()*1.01
    breakout = c15[i] > h15[max(0,i-5):i].max()
    if pullback and breakout and c15[i]>o15[i]:
        le[i]=True
rep('E3 Pullback + break above high', le, 4.0, 2.0)

# E4: Volume spike at 15m support + reversal
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    near_ema50 = l15[i] <= ema50_15[i]*1.005
    vol_spike = v15[i] > vsma20[i]*1.5
    rev = c15[i] > o15[i] and c15[i] > c15[i-1]
    if near_ema50 and vol_spike and rev:
        le[i]=True
rep('E4 Vol spike @ EMA50 + rev', le, 4.0, 2.0)

# E5: Previous 15m candle was a doji/pinbar at support + next green
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not trend_mask[i]: continue
    if i<2: continue
    # Previous candle: pin bar at low
    prev_body = abs(c15[i-1]-o15[i-1])
    prev_low_wick = min(c15[i-1],o15[i-1]) - l15[i-1]
    prev_pin = prev_low_wick > prev_body*2 and prev_low_wick > 0
    near_support = l15[i-1] <= swing_low_15[i-1]*1.01
    # Current: green candle
    if prev_pin and near_support and c15[i]>o15[i] and c15[i]>c15[i-1]:
        le[i]=True
rep('E5 PinBar@SwLow + next green', le, 4.0, 2.0)
