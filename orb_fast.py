#!/usr/bin/env python3
"""ORB fast test — specific anchor + range filter + fewer coins"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

DATA_PREV='/data/trading28/data/whale_15m_prev'
DATA_CUR='/data/trading28/data/whale_15m_1y'

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def load_one(dir_path,sym):
    p=os.path.join(dir_path,f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d=json.load(f)
    return (np.array(d['c'],float),np.array(d['h'],float),np.array(d['l'],float),np.array(d['o'],float),d.get('ts',[]))

def get_orb_signals(c,h,l,o,ts,anchor_hour,min_range_pct):
    """anchor_hour: UTC hour. min_range_pct: min ORB range as % of close"""
    n=len(c)
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
    except: return None,None
    
    long_e=np.zeros(n,bool); short_e=np.zeros(n,bool)
    orb_h=orb_l=0; orb_set=False; last_day=-1
    
    for i in range(200,n):
        d=idx[i].day; hr=idx[i].hour; mn=idx[i].minute
        
        # New day → find ORB candle
        if d!=last_day:
            last_day=d; orb_set=False
            # Look for first 15m candle at anchor_hour
            for j in range(i,min(i+96,n)):
                if idx[j].day!=d: break
                if idx[j].hour==anchor_hour and idx[j].minute==0:
                    orb_h=h[j]; orb_l=l[j]
                    orb_range=(orb_h-orb_l)/orb_l*100
                    if orb_range>=min_range_pct:
                        orb_set=True
                    break
        
        if not orb_set: continue
        
        # Only trade AFTER the ORB candle (not same candle)
        if hr==anchor_hour and mn==0: continue
        
        # Break signals
        if h[i]>=orb_h*1.0005 and c[i-1]<=orb_h:
            long_e[i]=True
        if l[i]<=orb_l*0.9995 and c[i-1]>=orb_l:
            short_e[i]=True
    
    return long_e, short_e

def sim(le,se,c,h,l,n,tp,sl):
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
        if not pos and cool==0:
            if le[i]: pos=1; ep=c[i]; side=1
            elif se[i]: pos=1; ep=c[i]; side=-1
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# Top liquid coins
LIQUID=['BTC','ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','DOT','LINK','MATIC','UNI','LTC','ATOM','ETC','FIL','APT','ARB','OP','NEAR','INJ','TIA','SUI','SEI','PEPE','WIF']

def run_test(anchor,min_rng,tp,sl,label):
    results_prev=[]; results_cur=[]
    for sym in LIQUID:
        pd_=load_one(DATA_PREV,sym); cd_=load_one(DATA_CUR,sym)
        if pd_ is None or cd_ is None: continue
        
        pc,ph,pl,po,pts=pd_; cc,ch,cl,co,cts=cd_
        
        ple,pse=get_orb_signals(pc,ph,pl,po,pts,anchor,min_rng)
        if ple is not None:
            r=sim(ple,pse,pc,ph,pl,len(pc),tp,sl)
            if r: r['sym']=sym; results_prev.append(r)
        
        cle,cse=get_orb_signals(cc,ch,cl,co,cts,anchor,min_rng)
        if cle is not None:
            r=sim(cle,cse,cc,ch,cl,len(cc),tp,sl)
            if r: r['sym']=sym; results_cur.append(r)
    
    for period,r in [('PREV',results_prev),('CUR',results_cur)]:
        if not r: continue
        tt=sum(x['t'] for x in r); tw=sum(x['w'] for x in r); tl=sum(x['l'] for x in r)
        tp_=sum(x['pnl'] for x in r); wr=tw/tt*100 if tt>0 else 0
        dd=np.mean([x['dd'] for x in r]); gr=sum(1 for x in r if x['pnl']>0)
        print(f"  {period}: {len(r)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")
    return results_prev, results_cur

# ── Test matrix ──
for anchor,alabel in [(13,'13:30 UTC (~NY open)'),(6,'06:30 UTC (~9:30 Jordan)')]:
    for min_rng in [0, 0.3, 0.5]:
        for tp,sl in [(3,1.5),(4,2),(5,2.5)]:
            print(f"\n{'='*70}")
            print(f"{alabel} | MinRange={min_rng}% | TP{tp}/SL{sl}")
            run_test(anchor,min_rng,tp,sl,'')
print("\nDone")
