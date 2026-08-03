#!/usr/bin/env python3
"""ADA 1m 7d — مراجعة + تصحيح — close-only SL/TP"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=7)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('ADA/USDT','1m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c)
idx=df.index

# SSL v2
period=10
sma_h=pd.Series(h).rolling(period).mean().values
sma_l=pd.Series(l_).rolling(period).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(period,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

# Whale — لا Look-ahead (كل القيم shift(1) للدخول)
LB=50
ln=pd.Series(l_).shift(1).rolling(LB).min().values    # shift(1)!
at_low=l_<=ln
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(at_low,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# Entry — مع shift: ندخل على الشمعة التالية للإشارة
le_v1=np.zeros(n,bool)  # الإصدار القديم (ممكن look-ahead)
le_v2=np.zeros(n,bool)  # الإصدار المصحح
for i in range(200,n):
    # v1: ندخل عند الإغلاق (نفس الشمعة)
    if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
        le_v1[i]=True
    
    # v2: ندخل على فتح الشمعة التالية (shift)
    if i>0 and ssl_c[i-1]==1 and wp_up[i-1] and wp[i-1]>wp[max(0,i-3)]*2 and wp[i-1]>0:
        le_v2[i]=True  # entry at open of NEXT candle

def sim_close_only(le, tp, sl):
    """SL/TP من الإغلاق الفعلي — close-only"""
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            # TP: close above target
            if c[i]>=ep*(1+tp/100):
                pnl=(c[i]/ep-1)*100-COMM*100
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
            # SL: close below stop
            elif c[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
        if not pos and cool==0 and le[i]:
            pos=1; ep=o[i]  # enter at OPEN
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    return t,cv,eq

def sim_high_low(le, tp, sl):
    """SL/TP من الهاي/لو — الطريقة القديمة"""
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=12
            elif l_[i]<=ep*(1-sl/100):
                t.append((c[i]/ep-1)*100-COMM*100); eq*=(1+((c[i]/ep-1)*100-COMM*100)/100); pos=0; cool=12
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    return t,cv,eq

p_change=(c[-1]-c[0])/c[0]*100

print(f'ADA 1m 7d | {n} candles | Δ {p_change:+.1f}%')
print(f'v1 (original): {le_v1.sum()} signals | v2 (shifted): {le_v2.sum()} signals')
print()

# Compare methods
for label, le in [('الإصدار القديم (مشبوه)', le_v1), ('مصحح shift(1) + close-only', le_v2)]:
    print(f'{"="*60}')
    print(f'🔍 {label}')
    print(f'{"TP/SL":>10} {"T":>4} {"WR":>6} {"DD":>6} {"Eq":>8} {"W/L":>8}')
    print('-'*45)
    
    for tp,sl in [(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
        if label=='مصحح shift(1) + close-only':
            tr,cv,eq=sim_close_only(le,tp,sl)
        else:
            tr,cv,eq=sim_high_low(le,tp,sl)
        
        if len(tr)<3: continue
        w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
        wr=len(w)/len(tr)*100
        dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
        ico='✅' if eq>CAP else '❌'
        print(f'{tp:.1f}%/{sl:.1f}%   {len(tr):>3} {wr:>5.1f}% {dd:>5.1f}% {ico}${eq-CAP:>+7.1f} {len(w)}W/{len(lo)}L')

# Manual audit: show first 3 trades for v1
print(f'\n{"="*60}')
print('🔎 تدقيق أول 5 صفقات (الإصدار القديم) — TP5/SL2.5:')
tr,_,_=sim_high_low(le_v1,5.0,2.5)
entries_v1=np.where(le_v1)[0]
for i in range(min(5,len(entries_v1))):
    ei=entries_v1[i]
    print(f'  #{i+1} ENTRY: {idx[ei]} @ ${c[ei]:.5f} C:{c[ei]:.5f} O:{o[ei]:.5f} H:{h[ei]:.5f} L:{l_[ei]:.5f}')

# Manual audit v2
print(f'\n🔎 تدقيق أول 5 صفقات (مصحح shift+close) — TP5/SL2.5:')
tr2,_,_=sim_close_only(le_v2,5.0,2.5)
entries_v2=np.where(le_v2)[0]
for i in range(min(5,len(entries_v2))):
    ei=entries_v2[i]
    print(f'  #{i+1} ENTRY: {idx[ei]} @ OPEN:{o[ei]:.5f} (prev close:{c[ei-1]:.5f})')

print('\n✅ تم')
