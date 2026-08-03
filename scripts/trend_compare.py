#!/usr/bin/env python3
"""
تبريد 12h + فلتر ترند 4h — اختبار
"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'
MAX_SLIPPAGE=1.5; COOLDOWN=48

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def sim(le, c, h, l_, n, tp, sl):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; crash=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                if raw<pnl: crash+=1
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq,crash

def test_coin(sym, trend_filter='none'):
    d=load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c); idx=d['ts']
    
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: return None
    
    # ── Trend filter from 4h data ──
    trend_ok=np.ones(n,bool)
    if trend_filter!='none':
        try:
            df=pd.DataFrame({'c':c},index=idx)
            c4h=df['c'].resample('4h').last().dropna().values
            e50_4h=ema(c4h,50); e200_4h=ema(c4h,200)
            e50_4h_aligned=np.zeros(n)
            for i in range(n):
                j=i//16
                if j<len(e50_4h): e50_4h_aligned[i]=e50_4h[j]
            e200_4h_aligned=np.zeros(n)
            for i in range(n):
                j=i//16
                if j<len(e200_4h): e200_4h_aligned[i]=e200_4h[j]
            
            if trend_filter=='ema_cross':
                trend_ok=(e50_4h_aligned>e200_4h_aligned)
            elif trend_filter=='price_above':
                # shift c4h to avoid look-ahead
                c4h_aligned=np.zeros(n)
                for i in range(n):
                    j=i//16
                    if j>0 and j<len(c4h): c4h_aligned[i]=c4h[j-1]  # shift(1)
                trend_ok=(c4h_aligned>e50_4h_aligned)
        except:
            trend_ok=np.ones(n,bool)
    
    best={'eq':0,'score':-999999}
    configs=[(30,3),(50,3),(70,3)]
    ssls=[5,10,20]
    tps=[(2,1),(3,1.5),(5,2.5)]
    
    for LB,sm in configs:
        ln=pd.Series(l_).shift(1).rolling(LB).min().values
        lc=np.zeros(n)
        for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
        sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
        hc=pd.Series(sc).rolling(LB).max().values
        sr=np.where(l_<=ln,(sc+hc*2)/3,0)
        wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
        wp_up=wp>np.roll(wp,1)
        
        for sp in ssls:
            sma_h=pd.Series(h).rolling(sp).mean().values
            sma_l=pd.Series(l_).rolling(sp).mean().values
            ssl_c=np.zeros(n,int)
            for i in range(sp,n):
                if h[i-1]>sma_h[i-1]: ssl_c[i]=1
                else: ssl_c[i]=-1
            
            le=np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and trend_ok[i]:
                    le[i]=True
            
            for tp,sl in tps:
                if le.sum()<5: continue
                tr,cv,eq,cr=sim(le,c,h,l_,n,tp,sl)
                if len(tr)<5: continue
                w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
                wr=len(w)/len(tr)*100
                dd_val=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
                score=(eq-CAP)-abs(dd_val)*15-cr*100+wr*1.5
                if score>best['score']:
                    best={'name':'W+SSL','LB':LB,'ssl':sp,'tp':tp,'sl':sl,
                          't':len(tr),'wr':wr,'dd':dd_val,'eq':eq,'crash':cr,'score':score}
    if best['eq']==0: return None
    return {'sym':sym,'best':best}

# ── Test 3 trend filters ──
print('🔄 مقارنة فلاتر الترند...')
coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])

for trend_name in ['none','ema_cross','price_above']:
    print(f'\n── {trend_name} ──')
    all_data=[]
    for i,sym in enumerate(coins):
        r=test_coin(sym, trend_name)
        if r: all_data.append(r)
    print(f'   {len(all_data)} عملة')
    
    filters={'min_wr':40,'max_dd':18,'min_trades':8,'max_crash':3}
    passed=[d for d in all_data if d['best']['eq']>CAP and d['best']['wr']>=filters['min_wr']
        and abs(d['best']['dd'])<=filters['max_dd'] and d['best']['t']>=filters['min_trades']
        and d['best']['crash']<=filters['max_crash']]
    
    if passed:
        avg_p=np.mean([d['best']['eq']-CAP for d in passed])
        avg_w=np.mean([d['best']['wr'] for d in passed])
        avg_d=np.mean([d['best']['dd'] for d in passed])
        total=sum(d['best']['eq']-CAP for d in passed)
        print(f'   ✅ {len(passed)} عملة | ربح +${total:.0f} | WR {avg_w:.1f}% | DD {avg_d:.1f}% | ربح/عملة +${avg_p:.0f}')
    else:
        print(f'   ❌ لا توجد')

print('\n✅ Done')
