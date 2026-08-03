#!/usr/bin/env python3
"""
منهجية يوسف + فلاتر اتجاه مختلفة — FET 3m 14d
اختبار 7 فلاتر اتجاه
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

def fetch(tf, days=14):
    ex=ccxt.binance({'timeout':15000})
    since=ex.parse8601((datetime.utcnow()-timedelta(days=days)).isoformat())
    all_c=[]
    while True:
        batch=ex.fetch_ohlcv('FET/USDT',tf,since=since,limit=1000)
        if not batch: break
        all_c.extend(batch)
        since=batch[-1][0]+1
        if len(batch)<1000: break
    df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms')
    df.set_index('ts',inplace=True); df.sort_index(inplace=True)
    return df

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

# Fetch
d3=fetch('3m',14)
c3=d3['close'].values; h3=d3['high'].values
l3=d3['low'].values; o3=d3['open'].values; n=len(c3); idx3=d3.index

d15=fetch('15m',14)
d1h=fetch('1h',14)
d4h=fetch('4h',30)

# Align higher TF to 3m index (shift(1) to avoid look-ahead)
def align(s, src_idx, tgt_idx):
    return pd.Series(s.values, index=src_idx).shift(1).reindex(tgt_idx, method='ffill').values

ema50_15=align(pd.Series(ema(d15['close'],50),index=d15.index),d15.index,idx3)
ema200_15=align(pd.Series(ema(d15['close'],200),index=d15.index),d15.index,idx3)
ema50_1h=align(pd.Series(ema(d1h['close'],50),index=d1h.index),d1h.index,idx3)
ema200_1h=align(pd.Series(ema(d1h['close'],200),index=d1h.index),d1h.index,idx3)
ema50_4h=align(pd.Series(ema(d4h['close'],50),index=d4h.index),d4h.index,idx3)
ema200_4h=align(pd.Series(ema(d4h['close'],200),index=d4h.index),d4h.index,idx3)
c_15=align(pd.Series(d15['close'],index=d15.index),d15.index,idx3)
c_1h=align(pd.Series(d1h['close'],index=d1h.index),d1h.index,idx3)
c_4h=align(pd.Series(d4h['close'],index=d4h.index),d4h.index,idx3)

# SSL on 3m
p=10
sma_h=pd.Series(h3).rolling(p).mean().values
sma_l=pd.Series(l3).rolling(p).mean().values
ssl_c3=np.zeros(n,int)
for i in range(p,n):
    if h3[i-1]>sma_h[i-1]: ssl_c3[i]=1
    else: ssl_c3[i]=-1

# Whale on 3m
LB=50
ln=pd.Series(l3).shift(1).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l3[i]-l3[i-1])/l3[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l3<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# Whale active zone
whale_active=np.zeros(n,bool)
for i in range(200,n):
    if wp_up[i] and wp[i]>0:
        for j in range(i,min(i+8,n)): whale_active[j]=True

# Base entry conditions (without trend filter)
ssl_blue=ssl_c3==1
base_le=np.zeros(n,bool)
base_sl=np.zeros(n)
for i in range(200,n):
    if whale_active[i] and ssl_blue[i] and c3[i]<c3[i-1] and c3[i]<o3[i]:
        sl10=l3[max(0,i-10):i+1].min()
        sd=(c3[i]-sl10)/c3[i]*100
        if 0.2<sd<3.0: base_le[i]=True; base_sl[i]=sl10

# ── Trend filters ──
filters={}
# F1: 15m EMA50 > EMA200
filters['15m EMA50>200']=np.array([not np.isnan(ema50_15[i]) and not np.isnan(ema200_15[i]) and ema50_15[i]>ema200_15[i] for i in range(n)])
# F2: 1h EMA50 > EMA200
filters['1h EMA50>200']=np.array([not np.isnan(ema50_1h[i]) and not np.isnan(ema200_1h[i]) and ema50_1h[i]>ema200_1h[i] for i in range(n)])
# F3: 4h EMA50 > EMA200
filters['4h EMA50>200']=np.array([not np.isnan(ema50_4h[i]) and not np.isnan(ema200_4h[i]) and ema50_4h[i]>ema200_4h[i] for i in range(n)])
# F4: 15m price > EMA50
filters['15m Price>EMA50']=np.array([not np.isnan(ema50_15[i]) and c_15[i]>ema50_15[i] for i in range(n)])
# F5: 1h price > EMA50
filters['1h Price>EMA50']=np.array([not np.isnan(ema50_1h[i]) and c_1h[i]>ema50_1h[i] for i in range(n)])
# F6: 4h price > EMA50
filters['4h Price>EMA50']=np.array([not np.isnan(ema50_4h[i]) and c_4h[i]>ema50_4h[i] for i in range(n)])
# F7: 15m+1h both up (EMA50>200)
filters['15m+1h ↑']=filters['15m EMA50>200'] & filters['1h EMA50>200']
# F8: no filter
filters['بدون فلتر']=np.ones(n,bool)

def sim(le, sl_px, tp_pct):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; sl=0
    for i in range(200,n):
        if pos:
            if c3[i]>=ep*(1+tp_pct/100): pnl=(c3[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
            elif c3[i]<=sl: pnl=(c3[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
        if not pos and le[i]: pos=1; ep=c3[i]; sl=sl_px[i]
        cv.append(eq)
    if pos: pnl=(c3[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq

print(f'FET 3m 14d | إشارات أساسية: {base_le.sum()}')
print(f'\n{"فلتر":<18} {"فعال%":>6} {"T":>4} {"WR":>6} {"سحب":>6} {"Eq":>9} {"W/L":>7}')
print('-'*65)

for fname, fmask in filters.items():
    le=base_le & fmask
    if le.sum()<3: 
        print(f'{fname:<18} {(fmask.sum()/n*100):>5.0f}% {"—":>4} {"—":>6} {"—":>6} {"—":>9}')
        continue
    
    best_eq=0; best_r=None
    for tp in [1.5, 2.0]:
        tr,cv,eq=sim(le, base_sl, tp)
        if eq>best_eq: best_eq=eq; best_r=(tp,tr,cv,eq)
    
    tp,tr,cv,eq=best_r
    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100 if tr else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='✅' if eq>CAP else '❌'
    print(f'{fname:<18} {(fmask.sum()/n*100):>5.0f}% {len(tr):>4} {wr:>5.1f}% {dd:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

print('\n✅ Done')
