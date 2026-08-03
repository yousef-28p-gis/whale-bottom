#!/usr/bin/env python3
"""
تبريد 12 ساعة + تحسين DD — اختبار سريع
"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'
MAX_SLIPPAGE=1.5; COOLDOWN=48  # 12 hours at 15m

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def sim(le, c, h, l_, n, tp, sl, cd=COOLDOWN):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; crash=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw, -sl*MAX_SLIPPAGE-COMM*100)
                if raw<pnl: crash+=1
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cd
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    return t,cv,eq,crash

def test_coin(sym):
    d=load(sym)
    if d is None or len(d['c'])<2000: return None
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: return None
    
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
            
            # W+SSL entry
            le=np.zeros(n,bool)
            for i in range(200,n):
                if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
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

# ── Run ──
print('🔄 تبريد 12 ساعة...')
coins=sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f!='_manifest.json'])
all_data=[]
for i,sym in enumerate(coins):
    r=test_coin(sym)
    if r: all_data.append(r)
    if (i+1)%30==0: print(f'  {i+1}/{len(coins)}...')

# Filters
filters={'min_wr':40,'max_dd':18,'min_trades':10,'max_crash':3}
passed=[d for d in all_data if d['best']['eq']>CAP and d['best']['wr']>=filters['min_wr']
    and abs(d['best']['dd'])<=filters['max_dd'] and d['best']['t']>=filters['min_trades']
    and d['best']['crash']<=filters['max_crash']]

print(f'\n✅ {len(all_data)} عملة | فلتر WR≥{filters["min_wr"]}% DD≤{filters["max_dd"]}% → {len(passed)} عملة\n')

if passed:
    avg_p=np.mean([d['best']['eq']-CAP for d in passed])
    avg_w=np.mean([d['best']['wr'] for d in passed])
    avg_d=np.mean([d['best']['dd'] for d in passed])
    total=sum(d['best']['eq']-CAP for d in passed)
    print(f'🟢 {len(passed)} عملة | ربح +${total:.0f} | متوسط WR {avg_w:.1f}% | DD {avg_d:.1f}% | ربح/عملة +${avg_p:.0f}')
    
    print(f'\n{"عملة":<12} {"T":>4} {"WR":>6} {"DD":>6} {"💥":>3} {"ربح":>8} {"LB/SSL":>7} {"TP/SL":>7}')
    print('-'*58)
    for d in sorted(passed, key=lambda x:-x['best']['eq'])[:25]:
        b=d['best']
        print(f'{d["sym"]:<12} {b["t"]:>4} {b["wr"]:>5.1f}% {b["dd"]:>5.1f}% {b["crash"]:>3} +${b["eq"]-CAP:>7.0f} {b["LB"]}/{b["ssl"]}  {b["tp"]}/{b["sl"]}')
    
    with open('/data/trading28/portfolio_12h_cooldown.json','w') as f:
        json.dump([{'sym':d['sym'],'LB':d['best']['LB'],'ssl':d['best']['ssl'],
                     'tp':d['best']['tp'],'sl':d['best']['sl'],'t':d['best']['t'],
                     'wr':d['best']['wr'],'dd':d['best']['dd'],'eq':d['best']['eq']} for d in passed],f)
    print(f'\n💾 Saved')
else:
    print('لا توجد عملات!')

print('\n✅ Done')
