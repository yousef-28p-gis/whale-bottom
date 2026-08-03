#!/usr/bin/env python3
"""Memory-efficient comparison — process one coin at a time"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def backtest_coin(c, h, l_, o, n, cfg, t4_arr, t1_arr):
    LB=cfg['LB']; sp=cfg['ssl']; tp=cfg['tp']; sl=cfg['sl']; sm=3
    ln=pd.Series(l_).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sma_h=pd.Series(h).rolling(sp).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(sp,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    
    results = {}
    for fname, filt in [('4h',t4_arr), ('4h+1h',t4_arr & t1_arr)]:
        le=np.zeros(n,bool)
        for i in range(200,n):
            if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0 and filt[i]:
                le[i]=True
        if le.sum()<3: continue
        
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
        if len(t)<5: continue
        w=[p for p in t if p>0]
        wr=len(w)/len(t)*100
        dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
        results[fname]={'t':len(t),'wr':wr,'dd':dd,'eq':eq,'pnl':eq-CAP}
    return results

# Load one coin at a time
with open('/data/trading28/final_bot_config.json') as f:
    configs = {r['sym']: r for r in json.load(f)}

summaries = {'cur_4h':[], 'cur_1h':[], 'prev_4h':[], 'prev_1h':[]}
common = 0

for sym in list(configs):
    cfg = configs[sym]
    p1 = f'/data/trading28/data/whale_15m_1y/{sym}.json'
    p2 = f'/data/trading28/data/whale_15m_prev/{sym}.json'
    if not os.path.exists(p1) or not os.path.exists(p2): continue
    common += 1
    
    for pname, ppath in [('cur',p1), ('prev',p2)]:
        with open(ppath) as fh: d = json.load(fh)
        c=np.array(d['c'],float); h=np.array(d['h'],float)
        l_=np.array(d['l'],float); o=np.array(d['o'],float)
        n=len(c); ts=d.get('ts',[])
        
        # Trend
        try:
            if ts and len(ts)==n:
                idx=pd.to_datetime(np.array(ts),unit='ms')
                df=pd.DataFrame({'c':c},index=idx)
                c4h=df['c'].resample('4h').last().dropna().values
                e50=ema(c4h,50); e200=ema(c4h,200)
                e50a=np.zeros(n); e200a=np.zeros(n)
                for i in range(n):
                    j=i//16
                    if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
                t4=e50a>e200a
                c1h=df['c'].resample('1h').last().dropna().values
                e20=ema(c1h,20); e50h=ema(c1h,50)
                e20a=np.zeros(n); e50a2=np.zeros(n)
                for i in range(n):
                    j=i//4
                    if j<len(e20): e20a[i]=e20[j]; e50a2[i]=e50h[j]
                t1=e20a>e50a2
            else:
                t4=np.ones(n,bool); t1=np.ones(n,bool)
        except:
            t4=np.ones(n,bool); t1=np.ones(n,bool)
        
        r=backtest_coin(c,h,l_,o,n,cfg,t4,t1)
        if r:
            if '4h' in r: summaries[f'{pname}_4h'].append(r['4h'])
            if '4h+1h' in r: summaries[f'{pname}_1h'].append(r['4h+1h'])
        
        del c,h,l_,o,d; gc.collect()
    
    if common % 20 == 0:
        print(f'  {common} coins done...', flush=True)

# Print
def show(label, items):
    if not items: return
    t=sum(i['t'] for i in items)
    w=sum(int(i['t']*i['wr']/100) for i in items)
    pnl=sum(i['pnl'] for i in items)
    wr=w/t*100 if t>0 else 0
    dd=np.mean([i['dd'] for i in items])
    green=sum(1 for i in items if i['pnl']>0)
    print(f'{label}: {len(items)}c {t}t WR={wr:.1f}% DD={dd:.1f}% ${pnl:+.0f} ✅{green}')

print(f'\nCommon coins: {common}')
for period in ['cur','prev']:
    print(f'\n--- {period} ---')
    show('  4h only', summaries[f'{period}_4h'])
    show('  4h+1h', summaries[f'{period}_1h'])

print(f'\n--- COMBINED ---')
show('  4h only', summaries['cur_4h']+summaries['prev_4h'])
show('  4h+1h', summaries['cur_1h']+summaries['prev_1h'])
print('Done')
