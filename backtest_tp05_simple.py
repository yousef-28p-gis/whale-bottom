#!/usr/bin/env python3
"""Test: Full position TP=0.5%, SL=L2-0.5%, no BE, no Half TP"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM = 0.20; CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH = 10; DEV = 1.0; D = DEPTH//2; CONFIRM = D
MAX_POS = 2; TIME_BARS = 120
TP_PCT = 0.5; SL_PCT = -0.5; DIST_FILTER = 0.5

def find_zpatterns(pv):
    pats=[]
    for i in range(len(pv)-3):
        p0,p1,p2,p3=pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1]; B=p2[1]-p1[1]; C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and 0.38<=B/A<=0.79 and p3[1]<p1[1]:
                pats.append((p0,p1,p2,p3))
    return pats

def simulate(close,high,low,coin):
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<4: return[]
    pats=find_zpatterns(pv); trades=[]
    for H1,L1,H2,L2 in pats:
        eb=L2[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L2[1])/L2[1]*100>DIST_FILTER: continue
        sl=L2[1]*(1+SL_PCT/100)
        if sl>=ep: continue
        tp=ep*(1+TP_PCT/100)
        for j in range(eb+1, min(n,eb+TIME_BARS+1)):
            if high[j]>=tp:
                trades.append({'coin':coin,'eb':eb,'exit':j,'type':'TP',
                    'pnl':round((tp/ep-1)*100-COMM,4),'bars':j-eb}); break
            if low[j]<=sl:
                xp=close[j]; trades.append({'coin':coin,'eb':eb,'exit':j,'type':'SL',
                    'pnl':round((xp/ep-1)*100-COMM,4),'bars':j-eb}); break
        else:
            jj=min(eb+TIME_BARS,n-1); xp=close[jj]
            trades.append({'coin':coin,'eb':eb,'exit':jj,'type':'TIME',
                'pnl':round((xp/ep-1)*100-COMM,4),'bars':jj-eb})
    return trades

with open('/data/trading28/config/shariah_coins.json') as f:
    sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

print(f'⏳ {len(COINS)}...', flush=True)
all_t=[]
for ci,cn in enumerate(COINS):
    fp=f'{DATA_DIR}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t.extend(simulate(df['close'].values,df['high'].values,df['low'].values,cn))
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)} — {len(all_t)}', flush=True)

all_t.sort(key=lambda t:t['eb'])
executed=[]; active=[]; skipped=0
for t in all_t:
    active=[a for a in active if a>t['eb']]
    if len(active)>=MAX_POS: skipped+=1; continue
    active.append(t['exit']); executed.append(t)

pnls=[t['pnl'] for t in executed]
wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
wr=len(wins)/len(pnls)*100 if pnls else 0
aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0

eq=CAPITAL; peq=CAPITAL; mdd=0; cons=0; maxc=0
for p in pnls:
    eq*=(1+0.10*p/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
    if p<=0: cons+=1; maxc=max(maxc,cons)
    else: cons=0

cs={}
for t in executed:
    cn=t['coin']
    if cn not in cs: cs[cn]={'w':0,'l':0,'net':0}
    if t['pnl']>0: cs[cn]['w']+=1
    else: cs[cn]['l']+=1
    cs[cn]['net']+=t['pnl']

tp_c=sum(1 for t in executed if t['type']=='TP')
sl_c=sum(1 for t in executed if t['type']=='SL')
ti_c=sum(1 for t in executed if t['type']=='TIME')

print(f'''
═══ TP=0.5% كامل | SL=L2-0.5% | بدون BE ═══

📋 {len(all_t):,} ⏭️ {skipped:,} → ✅ {len(executed):,}
🟢 {len(wins):,} 🔴 {len(losses):,} | WR: {wr:.1f}%
🟢 +{aw:+.2f}% 🔴 {al:+.2f}% | R:R: {aw/abs(al):.1f}x
📉 سحب: {mdd:.1f}% | أطول خسائر: {maxc}
🏦 $1,000 → ${eq:,.0f} (+{(eq/CAPITAL-1)*100:.1f}%)
🎯 TP:{tp_c} 🛑 SL:{sl_c} ⏱️ TIME:{ti_c}
🟢 عملات: {sum(1 for c in cs.values() if c['net']>0)} 🔴: {sum(1 for c in cs.values() if c['net']<=0)}
''')
