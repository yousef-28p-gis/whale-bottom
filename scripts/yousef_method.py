#!/usr/bin/env python3
"""
منهجية يوسف:
- الحوت يظهر (أعمدة خضرا) → ننتظر
- الأعمدة تبدأ تقصر (فقدان زخم)
- SSL أزرق
- ندخل على شمعة هابطة (حمرا) — الشمعة الثانية
- ستوب قريب تحت القاع السابق
- هدف قريب
"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# SSL v2
p=10
sma_h=pd.Series(h).rolling(p).mean().values
sma_l=pd.Series(l_).rolling(p).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(p,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

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

# ── منهجية يوسف ──
# 1. نحدد فترات ظهور الحوت (أعمدة خضرا متتالية)
# 2. ننتظر حتى تقصر الأعمدة
# 3. SSL أزرق
# 4. ندخل على شمعة هابطة
# 5. الستوب = أدنى قاع خلال فترة الحوت
# 6. الهدف = 1-2% (قريب)

le=np.zeros(n,bool)
sl_level=np.zeros(n)  # سعر الستوب لكل دخول
tp_level=np.zeros(n)

# تحديد مجموعات الحوت المتتالية
in_whale_zone=False
whale_start=0
whale_peaks=[]  # (start, peak_i, end) لكل موجة حوت

i=200
while i<n:
    if not in_whale_zone:
        # بداية موجة حوت: 3 أعمدة خضرا متتالية
        if i+2<n and wp_up[i] and wp_up[i+1] and wp_up[i+2] and wp[i]>0:
            in_whale_zone=True
            whale_start=i
            i+=3
            continue
        i+=1
    else:
        # داخل موجة حوت — ننتظر حتى تضعف (3 أعمدة حمرا أو صفر)
        if i+2<n:
            red_streak = (not wp_up[i]) and (not wp_up[i+1]) and (wp[i+1]==0 or wp[i+1]<wp[i])
            if red_streak:
                # موجة الحوت انتهت
                peak_i = whale_start + np.argmax(wp[whale_start:i])
                whale_peaks.append((whale_start, peak_i, i))
                in_whale_zone=False
        i+=1

print(f'موجات الحوت: {len(whale_peaks)}')

# لكل موجة حوت، نشوف إذا SSL أزرق وندخل على شمعة هابطة بعد القمة
for ws, peak, we in whale_peaks:
    if we>=n: continue
    # نتأكد SSL أزرق عند القمة
    if ssl_c[peak]!=1: continue
    
    # نبحث عن شمعة هابطة بعد القمة (السعر ينزل)
    for j in range(peak+1, min(peak+10, n)):
        if c[j] < c[j-1] and c[j] < o[j]:  # شمعة حمرا هابطة
            # الستوب = أدنى قاع خلال موجة الحوت + آخر 5 شمعات
            sl_lookback = max(0, ws-5)
            sl_low = l_[sl_lookback:j].min()
            sl_dist = (c[j] - sl_low) / c[j] * 100
            
            # هدف صغير: 1-2% أو قمة سابقة
            tp_dist = 1.5  # هدف %
            
            # الستوب لازم يكون معقول (مش بعيد)
            if sl_dist > 0 and sl_dist < 3.0:
                le[j]=True
                sl_level[j]=sl_low
                tp_level[j]=c[j]*(1+tp_dist/100)
            break

print(f'إشارات الدخول: {le.sum()}')

# ── Simulate ──
for tp_name, use_fixed_tp in [('هدف 1.5%', True), ('هدف 2%', True)]:
    tp_pct = 1.5 if '1.5' in tp_name else 2.0
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; sl_px=0; tp_px=0
    for i in range(200,n):
        if pos:
            if c[i] >= tp_px:
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
            elif c[i] <= sl_px:
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0
        if not pos and le[i]:
            pos=1; ep=c[i]
            sl_px=sl_level[i]
            tp_px=ep*(1+tp_pct/100) if use_fixed_tp else tp_level[i]
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='✅' if eq>CAP else '❌'
    print(f'\n{tp_name}: {len(t)}t WR{wr:.0f}% DD{dd:.1f}% {ico}${eq-CAP:+.1f} | {len(w)}W/{len(lo)}L | Avg SL dist: {np.mean([(c[i]-sl_level[i])/c[i]*100 for i in range(n) if le[i]]):.2f}%')

# ── Show trades ──
print(f'\n📋 تفاصيل الصفقات (هدف 2%):')
trades=[]; pos=0; ep=0; sl_px=0; ei=0
for i in range(200,n):
    if pos:
        if c[i]>=ep*(1+2/100):
            trades.append((ei,i,ep,c[i],'TP',sl_px))
            pos=0
        elif c[i]<=sl_px:
            trades.append((ei,i,ep,c[i],'SL',sl_px))
            pos=0
    if not pos and le[i]: pos=1; ep=c[i]; sl_px=sl_level[i]; ei=i

for i,(ei,xi,ep,xp,tt,_sl) in enumerate(trades):
    pnl=(xp/ep-1)*100-COMM*100
    sl_dist=(ep-_sl)/ep*100
    print(f'  {i+1}. {tt} | {idx[ei]} | دخول ${ep:.5f} | هدف ${ep*1.02:.5f} | ستوب ${_sl:.5f} ({sl_dist:.1f}%) | خرج ${xp:.5f} | {pnl:+.2f}%')

print(f'\n✅ {len(trades)} صفقة')
