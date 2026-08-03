#!/usr/bin/env python3
"""ORB — memory efficient: process one coin at a time"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

DATA_PREV='/data/trading28/data/whale_15m_prev'
DATA_CUR='/data/trading28/data/whale_15m_1y'

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def orb_signals(c,h,l,o,ts,tp,sl,anchor_utc_hour,filt):
    """Get ORB break signals. anchor_utc_hour: e.g. 0 for midnight, 13 for 1PM UTC"""
    n=len(c)
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
    except: return None,None
    
    # ORB: first 15m candle (minute=0) at or after anchor_hour each day
    orb_h=np.full(n,np.nan); orb_l=np.full(n,np.nan)
    orb_ready=np.zeros(n,bool)
    
    last_day=-1; orb_set=False; my_h=my_l=0
    for i in range(n):
        d=idx[i].day; hr=idx[i].hour; mn=idx[i].minute
        if d!=last_day:
            orb_set=False; last_day=d
        if not orb_set and hr>=anchor_utc_hour and mn==0:
            my_h=h[i]; my_l=l[i]; orb_set=True
        if orb_set and d==last_day:
            orb_h[i]=my_h; orb_l[i]=my_l
            # ready after the ORB candle closes
            if idx[i]>idx[i] if False else True:
                orb_ready[i]=True
    
    # Mark ready only AFTER ORB candle (not during it)
    for i in range(1,n):
        if not orb_ready[i]: continue
        if np.isnan(orb_h[i-1]):
            orb_ready[i]=False
    
    long_e=np.zeros(n,bool); short_e=np.zeros(n,bool)
    for i in range(200,n):
        if not orb_ready[i] or np.isnan(orb_h[i]): continue
        if h[i]>=orb_h[i]*1.001 and c[i-1]<=orb_h[i]: long_e[i]=True
        if l[i]<=orb_l[i]*0.999 and c[i-1]>=orb_l[i]: short_e[i]=True
    
    return long_e, short_e

def sim(le,se,c,h,l,n,tp,sl,filt):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(200,n):
        if pos:
            if side==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            else:
                if l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and filt[i]:
            if le[i]: pos=1; ep=c[i]; side=1
            elif se[i]: pos=1; ep=c[i]; side=-1
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

def load_one(dir_path,sym):
    p=os.path.join(dir_path,f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d=json.load(f)
    return (np.array(d['c'],float),np.array(d['h'],float),np.array(d['l'],float),np.array(d['o'],float),d.get('ts',[]))

def trend_filter(c,ts,n,tf):
    if tf=='none': return np.ones(n,bool)
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50=ema(c4h,50); e200=ema(c4h,200)
        e50a=np.zeros(n); e200a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
        f=e50a>e200a
        if tf=='4h+1h':
            c1h=df['c'].resample('1h').last().dropna().values
            e20=ema(c1h,20); e50h=ema(c1h,50)
            e20a=np.zeros(n); e50a2=np.zeros(n)
            for i in range(n):
                j=i//4
                if j<len(e20): e20a[i]=e20[j]; e50a2[i]=e50h[j]
            f=f&(e20a>e50a2)
        return f
    except: return np.ones(n,bool)

# Get common coins
prev_files=set(f.replace('.json','') for f in os.listdir(DATA_PREV) if f.endswith('.json') and f!='_manifest.json')
cur_files=set(f.replace('.json','') for f in os.listdir(DATA_CUR) if f.endswith('.json') and f!='_manifest.json')
common=sorted(prev_files & cur_files)
print(f"Common coins: {len(common)}")

for anchor_h,anchor_label in [(0,'00:00 UTC'),(13,'13:00 UTC NY open')]:
    print(f"\n{'='*70}")
    print(f"ANCHOR: {anchor_label} — ORB first 15m candle")
    print(f"{'='*70}")
    
    for tp,sl in [(3,1.5),(4,2),(5,2.5)]:
        print(f"\n--- TP={tp}% SL={sl}% ---")
        
        for tf in ['none']:
            results_prev=[]; results_cur=[]
            
            for sym in common:
                pd_=load_one(DATA_PREV,sym)
                cd_=load_one(DATA_CUR,sym)
                if pd_ is None or cd_ is None: continue
                
                pc,ph,pl,po,pts=pd_; cc,ch,cl,co,cts=cd_
                
                # PREV
                pf=trend_filter(pc,pts,len(pc),tf)
                ple,pse=orb_signals(pc,ph,pl,po,pts,tp,sl,anchor_h,pf)
                if ple is not None:
                    r=sim(ple,pse,pc,ph,pl,len(pc),tp,sl,pf)
                    if r: r['sym']=sym; results_prev.append(r)
                
                # CUR
                cf=trend_filter(cc,cts,len(cc),tf)
                cle,cse=orb_signals(cc,ch,cl,co,cts,tp,sl,anchor_h,cf)
                if cle is not None:
                    r=sim(cle,cse,cc,ch,cl,len(cc),tp,sl,cf)
                    if r: r['sym']=sym; results_cur.append(r)
            
            for label,res in [('PREV',results_prev),('CUR',results_cur)]:
                if not res: continue
                tt=sum(r['t'] for r in res); tw=sum(r['w'] for r in res)
                tl=sum(r['l'] for r in res); tp_=sum(r['pnl'] for r in res)
                wr=tw/tt*100 if tt>0 else 0
                dd=np.mean([r['dd'] for r in res])
                gr=sum(1 for r in res if r['pnl']>0)
                print(f"  {label}: {len(res):2d}Ⓜ️ {tt:4d}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} green={gr}")
            
            gc.collect()

print("\nDone")
