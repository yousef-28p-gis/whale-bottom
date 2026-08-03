#!/usr/bin/env python3
"""Ichimoku comprehensive grid: more TFs × more params × more TP/SL"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN={'15m':48,'30m':24,'1h':12,'2h':6,'4h':4,'8h':2,'12h':2,'1d':1}

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
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
        rule=f'{tf_min}min' if tf_min<1440 else '1D'
        r=df.resample(rule).agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values,r['h'].values,r['l'].values,r['o'].values
    except: return None

def ichimoku(c,h,l,o,tenkan,kijun,senkou,tp,sl,cooldown):
    n=len(c)
    if n<senkou+30: return None
    
    h_t=pd.Series(h).rolling(tenkan).max().values
    l_t=pd.Series(l).rolling(tenkan).min().values
    t_arr=(h_t+l_t)/2
    
    h_k=pd.Series(h).rolling(kijun).max().values
    l_k=pd.Series(l).rolling(kijun).min().values
    k_arr=(h_k+l_k)/2
    
    h_s=pd.Series(h).rolling(senkou).max().values
    l_s=pd.Series(l).rolling(senkou).min().values
    sb_raw=(h_s+l_s)/2; sa_raw=(t_arr+k_arr)/2
    
    shift=kijun
    sa=np.full(n,np.nan); sb=np.full(n,np.nan)
    for i in range(max(shift,senkou),n-shift):
        if i+shift<n: sa[i+shift]=sa_raw[i]; sb[i+shift]=sb_raw[i]
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(senkou+shift,n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        
        cloud_top=max(sa[i],sb[i]); cloud_bot=min(sa[i],sb[i])
        above=c[i]>cloud_top; below=c[i]<cloud_bot
        golden=t_arr[i]>k_arr[i] and t_arr[i-1]<=k_arr[i-1]
        death=t_arr[i]<k_arr[i] and t_arr[i-1]>=k_arr[i-1]
        
        if pos:
            if side==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
            else:
                if l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
        
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
timeframes = [
    ('30m',30),('1h',60),('2h',120),('4h',240),('8h',480),('12h',720),('1d',1440)
]
params = [
    (3,9,18,'Ultra 3/9/18'),
    (5,13,26,'Fast 5/13/26'),
    (7,22,44,'Crypto 7/22/44'),
    (9,26,52,'Standard 9/26/52'),
    (10,30,60,'Mid 10/30/60'),
    (13,34,68,'Slow 13/34/68'),
]
tp_sl = [(3,1.5),(4,2),(5,2.5),(6,3)]

print("=== ICHIMOKU FULL GRID ===")
for tf_name, tf_min in timeframes:
    cd_key = f'{tf_min}min' if tf_min<1440 else '1d'
    cool = COOLDOWN.get(cd_key,1)
    print(f"\n{'='*60}")
    print(f"⏱️ {tf_name} (cooldown={cool})")
    print(f"{'='*60}")
    
    best_combined = -99999
    best_result = None
    
    for tenkan,kijun,senkou,pname in params:
        for tp,sl in tp_sl:
            pr=[]; cr=[]
            for sym in LIQ:
                pd_=load(DP,sym); cd_=load(DC,sym)
                if pd_ is None or cd_ is None: continue
                for data,period,results in [(pd_,'PREV',pr),(cd_,'CUR',cr)]:
                    c,h,l,o,ts=data
                    if tf_min==15: c4,h4,l4,o4=c,h,l,o
                    else:
                        resampled=resample_tf(c,h,l,o,ts,tf_min)
                        if resampled is None: continue
                        c4,h4,l4,o4=resampled
                    r=ichimoku(c4,h4,l4,o4,tenkan,kijun,senkou,tp,sl,cool)
                    if r: r['sym']=sym; results.append(r)
            
            if not pr or not cr: continue
            prev_pnl=sum(x['pnl'] for x in pr)
            cur_pnl=sum(x['pnl'] for x in cr)
            combined=prev_pnl+cur_pnl
            
            # Track top 3 per TF
            if combined > best_combined:
                # Only show if both periods are positive
                pass
            
            prev_tt=sum(x['t'] for x in pr); prev_tw=sum(x['w'] for x in pr)
            cur_tt=sum(x['t'] for x in cr); cur_tw=sum(x['w'] for x in cr)
            prev_wr=prev_tw/prev_tt*100 if prev_tt>0 else 0
            cur_wr=cur_tw/cur_tt*100 if cur_tt>0 else 0
            prev_gr=sum(1 for x in pr if x['pnl']>0)
            cur_gr=sum(1 for x in cr if x['pnl']>0)
            
            # Only print if both periods positive AND WR>35%
            if prev_pnl>0 and cur_pnl>0 and prev_wr>35 and cur_wr>35:
                print(f"✅ {pname} TP{tp}/SL{sl}: PREV WR={prev_wr:.1f}% ${prev_pnl:+,.0f} g={prev_gr} | CUR WR={cur_wr:.1f}% ${cur_pnl:+,.0f} g={cur_gr} | Σ${combined:+,.0f}")
            
            # Track best
            if combined > best_combined:
                best_combined = combined
                best_result = (pname,tp,sl,prev_pnl,cur_pnl,prev_wr,cur_wr,prev_gr,cur_gr,prev_tt,cur_tt)
    
    if best_result:
        pname,tp,sl,pp,cp,pw,cw,pg,cg,pt,ct = best_result
        print(f"🏆 BEST: {pname} TP{tp}/SL{sl} | PREV WR={pw:.1f}% ${pp:+,.0f} g={pg} | CUR WR={cw:.1f}% ${cp:+,.0f} g={cg} | Σ${pp+cp:+,.0f}")
    gc.collect()

print("\nDone")
