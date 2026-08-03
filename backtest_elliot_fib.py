#!/usr/bin/env python3
"""Elliot 5-Wave + Fib Targets — Backtest"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; CAPITAL=1000; RISK=0.50
DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH=10; DEV=1.0; D=DEPTH//2; CONFIRM=D
MAX_POS=2; TIME_BARS=240; DIST_FILTER=0.5

# Configs to test
CONFIGS=[
    ('Fib382_full',  0.382, None,   False),  # Full TP at fib 0.382
    ('Fib618_full',  0.618, None,   False),  # Full TP at fib 0.618
    ('Fib1_full',    1.0,   None,   False),  # Full TP at fib 1.0
    ('Half382_618',  0.382, 0.618,  True),   # Half at 0.382, rest at 0.618 + BE
    ('Half382_1',    0.382, 1.0,    True),   # Half at 0.382, rest at 1.0 + BE
]

def find_5waves_down(pv):
    """Find completed downward 5-wave impulses"""
    pats=[]
    for i in range(len(pv)-5):
        p=pv[i:i+6]
        if [pt[2] for pt in p]!=['H','L','H','L','H','L']: continue
        H1=p[0]; L1=p[1]; H2=p[2]; L2=p[3]; H3=p[4]; L3=p[5]
        w1=H1[1]-L1[1]; w2=H2[1]-L1[1]; w3=H2[1]-L2[1]; w4=H3[1]-L2[1]; w5=H3[1]-L3[1]
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3[1]>=L1[1]: continue  # wave4 < wave1
        if L3[1]>=L2[1]: continue  # lower low
        if not (L1[1]<H1[1] and L2[1]<L1[1] and L3[1]<L2[1]): continue
        # Fibonacci validation
        ret2=w2/w1; ext3=w3/w1
        if not (0.33<=ret2<=0.82): continue  # broad fib range
        pats.append((H1,L1,H2,L2,H3,L3,w1))
    return pats

def simulate(close,high,low,coin):
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: return[]
    pats=find_5waves_down(pv)
    all_t=[]
    
    for H1,L1,H2,L2,H3,L3,wave_range in pats:
        eb=L3[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L3[1])/L3[1]*100>DIST_FILTER: continue
        
        sl=L3[1]*0.995  # 0.5% below wave5 low
        if sl>=ep: continue
        
        # Fib targets
        fib382=L3[1]+wave_range*0.382
        fib618=L3[1]+wave_range*0.618
        fib1  =L3[1]+wave_range*1.0
        
        # Check if targets make sense (> entry)
        if fib382<=ep: continue
        
        # For each config, simulate exit
        for cfg_name, tp_half_fib, tp_full_fib, use_be in CONFIGS:
            if tp_full_fib is None:
                # Full TP only (no half)
                target = L3[1]+wave_range*tp_half_fib
                if target<=ep: continue
                exit_type='TIME'; exit_pnl=0.0
                for j in range(eb+1, min(n,eb+TIME_BARS+1)):
                    if high[j]>=target:
                        exit_type='TP'; exit_pnl=(target/ep-1)*100-COMM; break
                    if low[j]<=sl:
                        exit_type='SL'; exit_pnl=(close[j]/ep-1)*100-COMM; break
                else:
                    jj=min(eb+TIME_BARS,n-1)
                    exit_pnl=(close[jj]/ep-1)*100-COMM
                    j=jj
                
                all_t.append({'cfg':cfg_name,'coin':coin,'eb':eb,'pnl':round(exit_pnl,4),'type':exit_type,'exit_bar':j})
            else:
                # Half TP + BE
                target_half=L3[1]+wave_range*tp_half_fib
                target_full=L3[1]+wave_range*tp_full_fib
                if target_half<=ep or target_full<=ep: continue
                
                half_exited=False; half1_pnl=0.0
                exit_type='TIME'; exit_pnl=0.0
                
                for j in range(eb+1, min(n,eb+TIME_BARS+1)):
                    bh=high[j]; bl=low[j]; bc=close[j]
                    
                    if not half_exited:
                        if bh>=target_half:
                            half1_pnl=(target_half/ep-1)*100-COMM/2
                            half_exited=True
                            if bh>=target_full:
                                h2=(target_full/ep-1)*100-COMM/2
                                exit_type='TP'; exit_pnl=(half1_pnl+h2)/2; break
                            continue
                        if bc<=sl:
                            exit_type='SL'; exit_pnl=(bc/ep-1)*100-COMM; break
                    
                    if half_exited:
                        if bh>=target_full:
                            h2=(target_full/ep-1)*100-COMM/2
                            exit_type='TP'; exit_pnl=(half1_pnl+h2)/2; break
                        if use_be and bc<=ep:
                            h2=(ep/ep-1)*100-COMM/2
                            exit_type='BE'; exit_pnl=(half1_pnl+h2)/2; break
                else:
                    jj=min(eb+TIME_BARS,n-1)
                    if half_exited:
                        h2=(close[jj]/ep-1)*100-COMM/2
                        exit_pnl=(half1_pnl+h2)/2
                    else:
                        exit_pnl=(close[jj]/ep-1)*100-COMM
                
                all_t.append({'cfg':cfg_name,'coin':coin,'eb':eb,'pnl':round(exit_pnl,4),'type':exit_type,'exit_bar':j})
    
    return all_t

# Load
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

print(f'⏳ Elliot 5-Wave + Fib — {len(COINS)}...', flush=True)
all_t={c[0]:[] for c in CONFIGS}
for ci,cn in enumerate(COINS):
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    trades=simulate(df['close'].values,df['high'].values,df['low'].values,cn)
    for t in trades: all_t[t['cfg']].append(t)
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)} — {sum(len(v) for v in all_t.values())}', flush=True)

# Per-config stats
print()
for cfg_name,_,_,_ in CONFIGS:
    trades=all_t[cfg_name]
    if not trades: print(f'{cfg_name}: 0'); continue
    
    # MAX_POS=2 — need exit bar for each trade
    # Add exit_bar to each trade first
    for t in trades:
        if 'exit_bar' not in t:
            t['exit_bar'] = t['eb'] + 100  # default
    
    trades.sort(key=lambda t:t['eb'])
    execd=[]; active=[]; skipped=0
    for t in trades:
        active=[a for a in active if a>t['eb']]
        if len(active)>=MAX_POS:
            skipped+=1; continue
        active.append(t['exit_bar'])
        execd.append(t)
    
    pnls=[t['pnl'] for t in execd]
    wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    wr=len(wins)/len(pnls)*100 if pnls else 0
    aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0
    
    eq=CAPITAL; peq=CAPITAL; mdd=0
    for p in pnls:
        eq*=(1+RISK*p/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
    
    tp_c=sum(1 for t in execd if t['type']=='TP')
    sl_c=sum(1 for t in execd if t['type']=='SL')
    be_c=sum(1 for t in execd if t['type']=='BE')
    
    print(f'{cfg_name}: {len(execd)}t | WR {wr:.1f}% | W {aw:+.2f}% L {al:+.2f}% | R:R {aw/abs(al):.1f}x | DD {mdd:.1f}% | ${CAPITAL}→${eq:,.0f} | TP{tp_c} SL{sl_c} BE{be_c}')
