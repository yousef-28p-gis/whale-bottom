#!/usr/bin/env python3
"""A2 — TP=HIGH, SL=CLOSE — 3-year FET 15m — with yearly breakdown"""
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

print('Fetching...')
d1=fetch('1d',DAYS); d4=fetch('4h',DAYS); d15=fetch('15m',DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
n15=len(c15); idx15=d15.index

c1d=pd.Series(d1['close'].shift(1).values,index=d1.index).reindex(idx15,method='ffill').values
o1d=pd.Series(d1['open'].shift(1).values,index=d1.index).reindex(idx15,method='ffill').values
c4h=pd.Series(d4['close'].shift(1).values,index=d4.index).reindex(idx15,method='ffill').values
o4h=pd.Series(d4['open'].shift(1).values,index=d4.index).reindex(idx15,method='ffill').values

ema20=pd.Series(c15).ewm(span=20,adjust=False).mean().values

le=np.zeros(n15,bool)
for i in range(200,n15):
    if not np.isnan(c1d[i]) and c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=ema20[i]*1.005 and c15[i]>ema20[i]*0.99 and c15[i]>o15[i]:
            le[i]=True

# TP=HIGH, SL=CLOSE
tp=4.0; sl=2.0
trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0; ei=0
trade_log=[]

for i in range(200,n15):
    if pos==1:
        if h15[i]>=ep*(1+tp/100):  # TP hit (HIGH)
            pnl=tp-COMM*100; trades.append(pnl); eq*=(1+pnl/100)
            trade_log.append({'yr':idx15[i].year,'pnl':pnl,'type':'TP','dur':i-ei})
            pos=0
        elif c15[i]<=ep*(1-sl/100):  # SL hit (CLOSE)
            pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100)
            trade_log.append({'yr':idx15[i].year,'pnl':pnl,'type':'SL','dur':i-ei})
            pos=0
    if pos==0 and le[i]: pos=1; ep=c15[i]; ei=i
    curve.append(eq)
if pos:
    pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100)
    trade_log.append({'yr':idx15[-1].year,'pnl':pnl,'type':'EOD','dur':n15-1-ei})

# Metrics
pnls=[t['pnl'] for t in trade_log]; nt=len(pnls)
w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
wr=len(w)/nt*100
aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
rr=abs(aw/al) if al else 99
dd=((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
sh=(pd.Series(curve).pct_change().dropna().mean()/pd.Series(curve).pct_change().dropna().std()*np.sqrt(365)) if pd.Series(curve).pct_change().std()>0 else 0
ann=(eq/CAP)**(365/DAYS)-1
tps=sum(1 for t in trade_log if t['type']=='TP')
sls=sum(1 for t in trade_log if t['type']=='SL')

print(f'\nA2 — TP=HIGH SL=CLOSE — TP{tp}/SL{sl} — 3 years')
print(f'{le.sum()} signals | {nt} trades | WR {wr:.1f}% | R:R {rr:.2f}x | DD {dd:.1f}%')
print(f'TP: {tps} | SL: {sls} | aW +{aw:.2f}% | aL {al:.2f}%')
print(f'Sh {sh:.2f} | Ann {ann*100:.0f}% | ${CAP} -> ${eq:.0f}')

# Yearly
print(f'\nYearly:')
for yr in sorted(set(t['yr'] for t in trade_log)):
    yt=[t for t in trade_log if t['yr']==yr]
    yp=[t['pnl'] for t in yt]; yw=[p for p in yp if p>0]
    y_wr=len(yw)/len(yp)*100 if yp else 0
    y_tp=sum(1 for t in yt if t['type']=='TP')
    y_sl=sum(1 for t in yt if t['type']=='SL')
    avg_dur=np.mean([t['dur'] for t in yt])
    print(f'  {yr}: {len(yt):>4d}t WR {y_wr:>5.1f}% TP{y_tp} SL{y_sl} Net{sum(yp):>+7.1f}% Dur{avg_dur:>5.0f}m')
