#!/usr/bin/env python3
"""Elliot 5-Wave + V-Shape exit (0.5%+1%) — direct comparison"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; CAPITAL=1000; RISK=0.50
DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH=10; DEV=1.0; D=DEPTH//2; CONFIRM=D
MAX_POS=2; TIME_BARS=120; DIST_FILTER=0.5
TP_PCT=1.0; HALF_TP_PCT=0.5; SL_PCT=-0.5

# Fib levels for validation
FIB_RETRACE = [0.382, 0.50, 0.618, 0.786]
FIB_EXTEND  = [1.0, 1.272, 1.382, 1.618]
FIB_TOL = 0.05

def near_fib(actual, fib_levels, tol=FIB_TOL):
    return any(abs(actual-f)<=tol for f in fib_levels)

def find_5waves_down(pv, use_fib=False):
    pats=[]
    for i in range(len(pv)-5):
        p=pv[i:i+6]
        if [pt[2] for pt in p]!=['H','L','H','L','H','L']: continue
        H1=p[0]; L1=p[1]; H2=p[2]; L2=p[3]; H3=p[4]; L3=p[5]
        w1=H1[1]-L1[1]; w2=H2[1]-L1[1]; w3=H2[1]-L2[1]; w4=H3[1]-L2[1]; w5=H3[1]-L3[1]
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3[1]>=L1[1]: continue
        if L3[1]>=L2[1]: continue
        
        if use_fib:
            ret2=w2/w1; ext3=w3/w1
            if not near_fib(ret2, FIB_RETRACE): continue
            if not near_fib(ext3, FIB_EXTEND): continue
        
        pats.append((H1,L1,H2,L2,H3,L3))
    return pats

def simulate(close,high,low,coin,use_fib):
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: return[]
    pats=find_5waves_down(pv,use_fib)
    trades=[]
    
    for H1,L1,H2,L2,H3,L3 in pats:
        eb=L3[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L3[1])/L3[1]*100>DIST_FILTER: continue
        
        sl=L3[1]*(1+SL_PCT/100)
        if sl>=ep: continue
        
        tp_full=ep*(1+TP_PCT/100); tp_half=ep*(1+HALF_TP_PCT/100); be=ep
        
        half_exited=False; half1_pnl=0.0
        exit_idx=eb; exit_type='TIME'; exit_pnl=0.0
        
        for j in range(eb+1, min(n,eb+TIME_BARS+1)):
            bh=high[j]; bl=low[j]; bc=close[j]
            
            if not half_exited:
                if bh>=tp_half:
                    half1_pnl=(tp_half/ep-1)*100-COMM/2; half_exited=True
                    if bh>=tp_full:
                        h2=(tp_full/ep-1)*100-COMM/2
                        exit_idx=j; exit_type='TP'; exit_pnl=(half1_pnl+h2)/2; break
                    continue
                if bc<=sl:
                    exit_idx=j; exit_type='SL'; exit_pnl=(bc/ep-1)*100-COMM; break
            
            if half_exited:
                if bh>=tp_full:
                    h2=(tp_full/ep-1)*100-COMM/2
                    exit_idx=j; exit_type='TP'; exit_pnl=(half1_pnl+h2)/2; break
                if bc<=be:
                    h2=(be/ep-1)*100-COMM/2
                    exit_idx=j; exit_type='BE'; exit_pnl=(half1_pnl+h2)/2; break
        else:
            jj=min(eb+TIME_BARS,n-1)
            if half_exited:
                h2=(close[jj]/ep-1)*100-COMM/2
                exit_pnl=(half1_pnl+h2)/2
            else:
                exit_pnl=(close[jj]/ep-1)*100-COMM
            exit_idx=jj
        
        trades.append({'coin':coin,'eb':eb,'exit_bar':exit_idx,'pnl':round(exit_pnl,4),'type':exit_type})
    return trades

# Load
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

for label,use_fib in [('Elliot (بدون فيبو)',False),('Elliot + فيبو',True)]:
    print(f'\n⏳ {label}...', flush=True)
    all_t=[]
    for ci,cn in enumerate(COINS):
        fp=f'{DATA}/{cn}.json'
        if not os.path.exists(fp): continue
        with open(fp) as f: raw=json.load(f)
        if len(raw)<200: continue
        df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        all_t.extend(simulate(df['close'].values,df['high'].values,df['low'].values,cn,use_fib))
        del df; gc.collect()
    print(f'  إشارات: {len(all_t)}', flush=True)
    
    all_t.sort(key=lambda t:t['eb'])
    execd=[]; active=[]; skipped=0
    for t in all_t:
        active=[a for a in active if a>t['eb']]
        if len(active)>=MAX_POS: skipped+=1; continue
        active.append(t['exit_bar']); execd.append(t)
    
    pnls=[t['pnl'] for t in execd]
    wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    wr=len(wins)/len(pnls)*100 if pnls else 0
    aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0
    
    eq=CAPITAL; peq=CAPITAL; mdd=0; cons=0; maxc=0
    for p in pnls:
        eq*=(1+RISK*p/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
        if p<=0: cons+=1; maxc=max(maxc,cons)
        else: cons=0
    
    tp_c=sum(1 for t in execd if t['type']=='TP')
    sl_c=sum(1 for t in execd if t['type']=='SL')
    be_c=sum(1 for t in execd if t['type']=='BE')
    ti_c=sum(1 for t in execd if t['type']=='TIME')
    
    print(f'  ✅ {len(execd)} | WR {wr:.1f}% | W {aw:+.2f}% L {al:+.2f}% | R:R {aw/abs(al):.1f}x | DD {mdd:.1f}% | ${CAPITAL}→${eq:,.0f} | خسائر متتالية {maxc} | TP{tp_c} SL{sl_c} BE{be_c} TIME{ti_c}')
