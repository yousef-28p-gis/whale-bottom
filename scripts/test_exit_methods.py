#!/usr/bin/env python3
"""مقارنة طرق الخروج: Hard SL | تريل | تعادل | جزئي"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'; COOLDOWN=48

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def get_entry(c,h,l_,n,LB=70,ssl_p=10):
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sma_h=pd.Series(h).rolling(ssl_p).mean().values; sma_l=pd.Series(l_).rolling(ssl_p).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(ssl_p,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    le=np.zeros(n,bool)
    for i in range(200,n):
        if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0: le[i]=True
    return le

def hard_sl(le,c,h,l_,n,tp,sl):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100): pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100; pnl=max(raw,-sl*1.5-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1; cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq

def trail_sl(le,c,h,l_,n,sl_pct):
    """تريل: SL يلحق السعر بعد كل قمة جديدة"""
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; trail_sl=0
    for i in range(200,n):
        if pos:
            if h[i]>ep: trail_sl=max(trail_sl, h[i]*(1-sl_pct/100))
            effective_sl=trail_sl if trail_sl>0 else ep*(1-sl_pct/100)
            if c[i]<=effective_sl:
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN; trail_sl=0
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]; trail_sl=0
        if not pos and cool>0: cool-=1; cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq

def be_sl(le,c,h,l_,n,tp,sl):
    """تعادل: SL ينتقل لنقطة الدخول بعد وصول +2%"""
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; moved=False
    for i in range(200,n):
        if pos:
            if not moved and h[i]>=ep*1.02: moved=True
            if h[i]>=ep*(1+tp/100): pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN; moved=False
            elif c[i]<=(ep if moved else ep*(1-sl/100)):
                if moved: pnl=-COMM*100
                else: raw=(c[i]/ep-1)*100-COMM*100; pnl=max(raw,-sl*1.5-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN; moved=False
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]; moved=False
        if not pos and cool>0: cool-=1; cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq

def m(tr,cv,eq):
    if len(tr)<3: return None
    w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return eq,wr,dd,len(w),len(lo)

# Load configs
with open('/data/trading28/final_bot_config.json') as f: configs=json.load(f)
N=20

methods={
    '🔥 Hard 5/2.5': lambda le,c,h,l_,n: hard_sl(le,c,h,l_,n,5,2.5),
    'Hard 3/1.5': lambda le,c,h,l_,n: hard_sl(le,c,h,l_,n,3,1.5),
    'Hard 2/1': lambda le,c,h,l_,n: hard_sl(le,c,h,l_,n,2,1),
    'تريل 2.5%': lambda le,c,h,l_,n: trail_sl(le,c,h,l_,n,2.5),
    'تريل 1.5%': lambda le,c,h,l_,n: trail_sl(le,c,h,l_,n,1.5),
    'تعادل TP5': lambda le,c,h,l_,n: be_sl(le,c,h,l_,n,5,2.5),
    'تعادل TP3': lambda le,c,h,l_,n: be_sl(le,c,h,l_,n,3,1.5),
}

print(f'🔄 مقارنة طرق الخروج — {N} عملة...\n')
results={name:{'eq':[],'wr':[],'dd':[],'t':[]} for name in methods}

for cfg in configs[:N]:
    d=load(cfg['sym'])
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    try:
        df=pd.DataFrame({'c':c},index=d['ts']); c4h=df['c'].resample('4h').last().dropna().values
        e50=pd.Series(c4h).ewm(span=50,adjust=False).mean().values
        e200=pd.Series(c4h).ewm(span=200,adjust=False).mean().values
        e50_a=np.zeros(n); e200_a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50): e50_a[i]=e50[j]; e200_a[i]=e200[j]
        trend_ok=e50_a>e200_a
    except: trend_ok=np.ones(n,bool)
    le=get_entry(c,h,l_,n,cfg['LB'],cfg['ssl']); le=le&trend_ok
    
    for name,fn in methods.items():
        tr,cv,eq=fn(le,c,h,l_,n)
        r=m(tr,cv,eq)
        if r: results[name]['eq'].append(r[0]); results[name]['wr'].append(r[1]); results[name]['dd'].append(r[2]); results[name]['t'].append(len(tr))

print(f'{"الطريقة":<18} {"✅":>4} {"إجمالي":>9} {"WR":>6} {"DD":>6} {"T/ع":>6}')
print('-'*55)
for name, r in results.items():
    if not r['eq']: continue
    total=sum(r['eq'])-CAP*len(r['eq'])
    avg_wr=np.mean(r['wr']); avg_dd=np.mean(r['dd']); avg_t=np.mean(r['t'])
    ico='✅' if total>0 else '❌'
    print(f'{name:<18} {len(r["eq"]):>4} {ico}${total:>+8.0f} {avg_wr:>5.1f}% {avg_dd:>5.1f}% {avg_t:>5.1f}')
print('\n✅ Done')
