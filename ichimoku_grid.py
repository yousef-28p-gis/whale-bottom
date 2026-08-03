#!/usr/bin/env python3
"""Ichimoku grid search — different params × different timeframes"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

DP='/data/trading28/data/whale_15m_prev'
DC='/data/trading28/data/whale_15m_1y'
LIQ=['BTC','ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','DOT','LINK',
     'MATIC','UNI','LTC','ATOM','ETC','FIL']

def load(d,s):
    p=os.path.join(d,f'{s}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j=json.load(f)
    return (np.array(j['c'],float),np.array(j['h'],float),np.array(j['l'],float),np.array(j['o'],float),j.get('ts',[]))

def resample_tf(c,h,l,o,ts,tf_min):
    """Resample 15m data to target timeframe"""
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
        rule=f'{tf_min}min'
        r=df.resample(rule).agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values,r['h'].values,r['l'].values,r['o'].values
    except: return None

def ichimoku_test(c,h,l,o,tenkan_p,kijun_p,senkou_p,tp,sl):
    """Ichimoku: Buy on golden cross + above cloud, Sell on death cross + below cloud"""
    n=len(c)
    if n<senkou_p+30: return None
    
    # Tenkan
    h_t=pd.Series(h).rolling(tenkan_p).max().values
    l_t=pd.Series(l).rolling(tenkan_p).min().values
    tenkan=(h_t+l_t)/2
    
    # Kijun
    h_k=pd.Series(h).rolling(kijun_p).max().values
    l_k=pd.Series(l).rolling(kijun_p).min().values
    kijun=(h_k+l_k)/2
    
    # Senkou B
    h_s=pd.Series(h).rolling(senkou_p).max().values
    l_s=pd.Series(l).rolling(senkou_p).min().values
    sb_raw=(h_s+l_s)/2
    
    # Senkou A = (Tenkan+Kijun)/2 shifted forward kijun periods
    sa_raw=(tenkan+kijun)/2
    
    # Shift forward
    shift=kijun_p
    sa=np.full(n,np.nan); sb=np.full(n,np.nan)
    for i in range(shift,n-shift):
        if i+shift<n: sa[i+shift]=sa_raw[i]
    for i in range(senkou_p,n-shift):
        if i+shift<n: sb[i+shift]=sb_raw[i]
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(senkou_p+shift,n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        
        cloud_top=max(sa[i],sb[i]); cloud_bot=min(sa[i],sb[i])
        above=c[i]>cloud_top; below=c[i]<cloud_bot
        golden=tenkan[i]>kijun[i] and tenkan[i-1]<=kijun[i-1]
        death=tenkan[i]<kijun[i] and tenkan[i-1]>=kijun[i-1]
        
        if pos:
            if side==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif death and not above:  # Exit on reverse cross if not in uptrend
                    pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            else:
                if l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif golden and not below:
                    pnl=(1-c[i]/ep)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        
        if not pos and cool==0:
            if above and golden: pos=1; ep=c[i]; side=1
            elif below and death: pos=1; ep=c[i]; side=-1
        
        if not pos and cool>0: cool-=1
        cv.append(eq)
    
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    
    if len(t)<3: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ── Grid ──
params = [
    (5,13,26, 'Fast 5/13/26'),
    (9,26,52, 'Standard 9/26/52'),
    (13,34,68, 'Slow 13/34/68'),
    (7,22,44, 'Crypto 7/22/44'),
]
timeframes = [(15,'15m'),(60,'1h'),(240,'4h')]

print("=== ICHIMOKU GRID SEARCH ===")
for tf_min,tf_name in timeframes:
    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_name}")
    print(f"{'='*60}")
    
    for tp,sl in [(3,1.5),(4,2),(5,2.5)]:
        best_combined = -99999
        best_config = None
        
        for tenkan,kijun,senkou, pname in params:
            pr=[]; cr=[]
            for sym in LIQ:
                pd_=load(DP,sym); cd_=load(DC,sym)
                if pd_ is None or cd_ is None: continue
                
                for data,period,results in [(pd_,'PREV',pr),(cd_,'CUR',cr)]:
                    c,h,l,o,ts=data
                    if tf_min==15:
                        c4,h4,l4,o4=c,h,l,o
                    else:
                        resampled=resample_tf(c,h,l,o,ts,tf_min)
                        if resampled is None: continue
                        c4,h4,l4,o4=resampled
                    
                    r=ichimoku_test(c4,h4,l4,o4,tenkan,kijun,senkou,tp,sl)
                    if r: r['sym']=sym; results.append(r)
            
            if not pr or not cr: continue
            prev_pnl=sum(x['pnl'] for x in pr)
            cur_pnl=sum(x['pnl'] for x in cr)
            combined=prev_pnl+cur_pnl
            
            if combined > best_combined:
                best_combined=combined
                best_config=(pname,pr,cr)
        
        if best_config:
            pname,pr,cr=best_config
            for label,res in [('PREV',pr),('CUR',cr)]:
                tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
                tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
                wr=tw/tt*100 if tt>0 else 0
                dd=np.mean([x['dd'] for x in res]); gr=sum(1 for x in res if x['pnl']>0)
                print(f"  {pname} TP{tp}/SL{sl} {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")
        else:
            print(f"  TP{tp}/SL{sl}: no trades")

print("\nDone")
