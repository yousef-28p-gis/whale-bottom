#!/usr/bin/env python3
"""Multi-TF — Strong Trend Filters — FET 15m — 3 years"""
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
d1=fetch('1d',DAYS); d4=fetch('4h',DAYS); d15=fetch('15m',DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
n15=len(c15); idx15=d15.index

def align_shift(hi,idx15):
    return pd.Series(hi.shift(1).values,index=hi.index).reindex(idx15,method='ffill').values

c1d=align_shift(d1['close'],idx15); o1d=align_shift(d1['open'],idx15)
h1d=align_shift(d1['high'],idx15); l1d=align_shift(d1['low'],idx15)
c4h=align_shift(d4['close'],idx15); o4h=align_shift(d4['open'],idx15)
h4h=align_shift(d4['high'],idx15); l4h=align_shift(d4['low'],idx15)

# Trend indicators on daily
ema50_d = align_shift(ema(d1['close'],50),idx15)
ema200_d = align_shift(ema(d1['close'],200),idx15)

# 15m EMA for bounce detection
ema20_15 = pd.Series(c15).ewm(span=20,adjust=False).mean().values

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
    dr=pd.Series(cv).pct_change().dropna()
    sh=(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
    ann=(eq/CAP)**(365/DAYS)-1
    tps=sum(1 for p in tr if p>0)
    sls=sum(1 for p in tr if p<=0)
    ico='+' if eq>CAP else '-'
    print(f'{name:<40} {le.sum():>5}s {len(tr):>4d}t WR{wr:>5.1f}% R:R{aw/abs(al) if al else 99:>5.2f}x DD{dd:>5.1f}% {ico}${eq-1000:>+8.0f} Sh{sh:.2f} TP{tps} SL{sls}')

# === ENTRY: EMA20 bounce ===
def ema20_entry(c15,o15,ema20_15,n15):
    le=np.zeros(n15,bool)
    for i in range(200,n15):
        if c15[i]<=ema20_15[i]*1.005 and c15[i]>ema20_15[i]*0.99 and c15[i]>o15[i]:
            le[i]=True
    return le

print(f'\n{"="*85}')
print(f'Multi-TF Strong Trend Filters — FET 15m — 3 years')
print(f'{"="*85}')
print(f'{"Filter":<40} {"Sigs":>5} {"T":>4} {"WR":>6} {"R:R":>5} {"DD":>6} {"Profit":>9} {"Sh":>5}')
print('-'*80)

# F0: No filter (baseline)
le=ema20_entry(c15,o15,ema20_15,n15)
rep('F0 No filter (just EMA20 bounce)', le, 4.0, 2.0)

# F1: Daily EMA50 > EMA200 (golden cross zone)
for i in range(200,n15):
    if np.isnan(ema50_d[i]) or np.isnan(ema200_d[i]): le[i]=False
    elif ema50_d[i] <= ema200_d[i]: le[i]=False
rep('F1 D:EMA50>EMA200', le, 4.0, 2.0)

# F2: Price > EMA50 on daily AND 4h
le=ema20_entry(c15,o15,ema20_15,n15)
for i in range(200,n15):
    if np.isnan(ema50_d[i]): le[i]=False
    elif c1d[i] <= ema50_d[i]: le[i]=False
    elif c4h[i] <= align_shift(ema(d4['close'],50),idx15)[i]: le[i]=False
rep('F2 Price>EMA50 D+4h', le, 4.0, 2.0)

# F3: Daily HH/HL structure (higher high and higher low on daily)
le=ema20_entry(c15,o15,ema20_15,n15)
for i in range(200,n15):
    if np.isnan(h1d[i]): le[i]=False
    # Last daily high > previous daily high + last daily low > previous daily low
    elif not (h1d[i] > h1d[max(0,i-96)] and l1d[i] > l1d[max(0,i-96)]): le[i]=False
rep('F3 D:HH+HL structure', le, 4.0, 2.0)

# F4: Daily close above previous daily high (strong bullish)
le=ema20_entry(c15,o15,ema20_15,n15)
for i in range(200,n15):
    if np.isnan(c1d[i]): le[i]=False
    elif c1d[i] <= h1d[max(0,i-96)]: le[i]=False  # close > previous day high
rep('F4 D:Close>PrevHigh', le, 4.0, 2.0)

# F5: Combo F1+F2 (EMA50>EMA200 + Price>EMA50 on both)
le=ema20_entry(c15,o15,ema20_15,n15)
ema50_4h_aligned = align_shift(ema(d4['close'],50),idx15)
for i in range(200,n15):
    if np.isnan(ema50_d[i]) or np.isnan(ema200_d[i]): le[i]=False
    elif ema50_d[i]<=ema200_d[i] or c1d[i]<=ema50_d[i] or c4h[i]<=ema50_4h_aligned[i]: le[i]=False
rep('F5 F1+F2 (EMA+Price)', le, 4.0, 2.0)

# F6: F5 + 4h also HH/HL
le=ema20_entry(c15,o15,ema20_15,n15)
ema50_4h_aligned = align_shift(ema(d4['close'],50),idx15)
for i in range(200,n15):
    if np.isnan(ema50_d[i]): le[i]=False
    elif ema50_d[i]<=ema200_d[i] or c1d[i]<=ema50_d[i] or c4h[i]<=ema50_4h_aligned[i]: le[i]=False
    elif not (h4h[i]>h4h[max(0,i-24)] and l4h[i]>l4h[max(0,i-24)]): le[i]=False
rep('F6 F5+4h HH/HL', le, 4.0, 2.0)

# F7: F5 + daily close > prev high
le=ema20_entry(c15,o15,ema20_15,n15)
ema50_4h_aligned = align_shift(ema(d4['close'],50),idx15)
for i in range(200,n15):
    if np.isnan(ema50_d[i]): le[i]=False
    elif ema50_d[i]<=ema200_d[i] or c1d[i]<=ema50_d[i] or c4h[i]<=ema50_4h_aligned[i]: le[i]=False
    elif c1d[i] <= h1d[max(0,i-96)]: le[i]=False
rep('F7 F5+D Close>PrevH', le, 4.0, 2.0)
