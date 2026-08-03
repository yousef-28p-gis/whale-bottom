#!/usr/bin/env python3
"""
منهجية يوسف — نسخة مخففة:
- موجة حوت تظهر (عمودين خضرا متتاليين)
- SSL أزرق في المنطقة
- دخول على أي شمعة هابطة بعد بداية الموجة
- ستوب = أدنى قاع آخر 10 شمعات
- هدف صغير 1.5-2%
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=14)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('FET/USDT','3m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c); idx=df.index

# SSL
p=10
sma_h=pd.Series(h).rolling(p).mean().values
sma_l=pd.Series(l_).rolling(p).mean().values
ssl_c=np.zeros(n,int)
for i in range(p,n):
    if h[i-1]>sma_h[i-1]: ssl_c[i]=1
    else: ssl_c[i]=-1

# Whale
LB=50
ln=pd.Series(l_).shift(1).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# ── تحديد موجات الحوت النشطة ──
# أي منطقة فيها عمودين خضرا متتاليين = موجة حوت نشطة
whale_active=np.zeros(n,bool)
for i in range(200,n):
    if wp_up[i] and wp[i]>0:
        # تفعيل المنطقة: 5 شمعات بعد آخر عمود أخضر
        for j in range(i, min(i+8, n)):
            whale_active[j]=True

# SSL أزرق
ssl_blue=ssl_c==1

# دخول: whale_active + SSL أزرق + شمعة هابطة (إغلاق < إغلاق سابق)
le=np.zeros(n,bool)
sl_px=np.zeros(n)
for i in range(200,n):
    if whale_active[i] and ssl_blue[i] and c[i] < c[i-1] and c[i] < o[i]:
        # الستوب = أدنى قاع آخر 10 شمعات (أو 0.5% أيهما أقرب)
        sl10=l_[max(0,i-10):i+1].min()
        sl_dist=(c[i]-sl10)/c[i]*100
        if sl_dist > 0.2 and sl_dist < 3.0:
            le[i]=True
            sl_px[i]=sl10

print(f'FET 3m 14d | {n} candles | موجات نشطة: {whale_active.sum()/n*100:.0f}%')
print(f'SSL أزرق: {ssl_blue.sum()/n*100:.0f}% | إشارات: {le.sum()}')
print(f'{"هدف":>8} {"T":>4} {"WR":>6} {"سحب":>6} {"Eq":>9} {"W/L":>8} {"متوسط SL":>8}')
print('-'*60)

for tp_pct in [1.0, 1.5, 2.0, 2.5]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; sl=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+tp_pct/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
            elif c[i]<=sl:
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
        if not pos and le[i]: pos=1; ep=c[i]; sl=sl_px[i]
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    avg_sl=np.mean([(c[i]-sl_px[i])/c[i]*100 for i in range(n) if le[i]])
    ico='✅' if eq>CAP else '❌'
    print(f'{tp_pct:.1f}%     {len(t):>4} {wr:>5.1f}% {dd:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L {avg_sl:>7.2f}%')

# Best trades detail
print(f'\n📋 صفقات (هدف 2%):')
trades=[]; pos=0; ep=0; sl=0; ei=0
for i in range(200,n):
    if pos:
        if c[i]>=ep*(1+2/100): trades.append((ei,i,ep,c[i],'TP',sl)); pos=0
        elif c[i]<=sl: trades.append((ei,i,ep,c[i],'SL',sl)); pos=0
    if not pos and le[i]: pos=1; ep=c[i]; sl=sl_px[i]; ei=i

for i,(ei,xi,ep,xp,tt,sl_v) in enumerate(trades):
    pnl=(xp/ep-1)*100-COMM*100
    print(f'  {i+1}. {tt} | {idx[ei]} → {idx[xi]} | ${ep:.5f}→${xp:.5f} | SL:${sl_v:.5f} | {pnl:+.2f}% | {(xi-ei)*3}min')

print(f'\n✅ {len(trades)} trades')
