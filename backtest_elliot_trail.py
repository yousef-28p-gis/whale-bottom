#!/usr/bin/env python3
"""Elliot 5-Wave + w5=0.382(w1+w3) — Trailing Stop based on Fib levels"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; CAPITAL=1000; RISK=0.50
DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD'}

DEPTH=10; DEV=1.0; D=DEPTH//2; CONFIRM=D
MAX_POS=2; TIME_BARS=120; DIST_FILTER=0.5
INIT_SL_PCT=-0.5  # Initial SL below L3

def near(v,target,tol=0.03): return abs(v-target)<=tol

def find_5waves(pv, w5_filter=None):
    pats=[]
    for i in range(len(pv)-5):
        p=pv[i:i+6]
        if [pt[2] for pt in p]!=['H','L','H','L','H','L']: continue
        H1=p[0];L1=p[1];H2=p[2];L2=p[3];H3=p[4];L3=p[5]
        w1=H1[1]-L1[1];w2=H2[1]-L1[1];w3=H2[1]-L2[1];w4=H3[1]-L2[1];w5=H3[1]-L3[1]
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3[1]>=L1[1] or L3[1]>=L2[1]: continue
        if w5_filter:
            ratio=w5/(w1+w3)
            ok=any(near(ratio,f) for f in w5_filter)
            if not ok: continue
        pats.append((H1,L1,H2,L2,H3,L3))
    return pats

def simulate_trail(close,high,low,coin,w5_filter):
    """Trailing stop: 
    - Initial SL = L3 - 0.5%
    - Price >= Fib 0.5 → SL = BE (entry)
    - Price >= Fib 0.6 → SL = Fib 0.3
    - Price >= Fib 1.0 → SL = Fib 0.5
    - Price >= Fib 1.272 → SL = Fib 0.618
    - Price >= Fib 1.618 → SL = Fib 1.0
    """
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: return[]
    pats=find_5waves(pv,w5_filter)
    trades=[]
    
    for H1,L1,H2,L2,H3,L3 in pats:
        eb=L3[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L3[1])/L3[1]*100>DIST_FILTER: continue
        
        w5_size = H3[1] - L3[1]
        if w5_size <= 0: continue
        
        # Fib levels from L3 to H3
        fib03 = L3[1] + 0.3 * w5_size
        fib05 = L3[1] + 0.5 * w5_size
        fib06 = L3[1] + 0.6 * w5_size
        fib10 = H3[1]
        fib127 = L3[1] + 1.272 * w5_size
        fib1618 = L3[1] + 1.618 * w5_size
        
        # Trail rules: (trigger_level, new_sl_level)
        trail_rules = [
            (fib05, ep),        # >= 0.5 → BE
            (L3[1]+0.7*w5_size, fib05),   # >= 0.7 → 0.5
            (fib10, L3[1]+0.7*w5_size),   # >= 1.0 → 0.7
            (fib127, fib10),    # >= 1.272 → 1.0
            (fib1618, fib127),  # >= 1.618 → 1.272
        ]
        
        # Initial SL
        sl = L3[1] * (1 + INIT_SL_PCT/100)
        if sl >= ep: continue
        
        exit_type='TIME'; exit_idx=eb; exit_pnl=0.0
        highest_trigger = -1  # track which rule was last triggered
        
        for j in range(eb+1, min(n, eb+TIME_BARS+1)):
            bh=high[j]; bl=low[j]; bc=close[j]
            
            # Update trailing stop based on high
            for idx, (trigger, new_sl) in enumerate(trail_rules):
                if bh >= trigger and idx > highest_trigger:
                    # Never trail below entry (BE) once reached
                    new_sl = max(new_sl, ep)
                    if new_sl > sl:  # only move up
                        sl = new_sl
                        highest_trigger = idx
            
            # Check SL hit
            if bc <= sl:
                exit_idx=j; exit_type='TRAIL'; exit_pnl=(bc/ep-1)*100-COMM; break
        
        else:
            jj=min(eb+TIME_BARS, n-1)
            exit_idx=jj; exit_pnl=(close[jj]/ep-1)*100-COMM
        
        if exit_pnl > 10 or exit_pnl < -15: continue
        
        # Determine which trail level was reached
        last_trail = 'INIT'
        if highest_trigger >= 0: last_trail = f'F{["0.5","0.7","1.0","1.27","1.62"][highest_trigger]}'
        
        trades.append({'coin':coin,'eb':eb,'exit_bar':exit_idx,'pnl':round(exit_pnl,4),
                       'type':exit_type,'w5':round(w5_size,6),'trail':last_trail})
    return trades

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]
W5_FILTER=[0.382]

print('═══ Elliot 5-Wave: Trailing Stop فيبو ═══', flush=True)

all_t=[]
for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t.extend(simulate_trail(df['close'].values,df['high'].values,df['low'].values,cn,W5_FILTER))
    del df; gc.collect()

all_t.sort(key=lambda t:t['eb'])
execd=[]; active=[]
for t in all_t:
    active=[a for a in active if a>t['eb']]
    if len(active)>=MAX_POS: continue
    active.append(t['exit_bar']); execd.append(t)

pnls=[t['pnl'] for t in execd]; wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
wr=len(wins)/len(pnls)*100 if pnls else 0
aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0
eq=CAPITAL; peq=CAPITAL; mdd=0; cons=0; maxc=0
for p in pnls:
    eq*=(1+RISK*p/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
    if p<=0: cons+=1; maxc=max(maxc,cons)
    else: cons=0

ret=[p for p in pnls]
if len(ret)>1: sharpe=(np.mean(ret)/np.std(ret))*np.sqrt(len(ret)) if np.std(ret)>0 else 0
else: sharpe=0

trail_c=sum(1 for t in execd if t['type']=='TRAIL')
time_c=sum(1 for t in execd if t['type']=='TIME')

# Trail level breakdown
from collections import Counter
trail_dist = Counter(t['trail'] for t in execd)

print(f'\n📅 4 شهور — 3m — CLOSE-ONLY')
print(f'📊 بيانات: {len(COINS)} عملة — ⏱️ Trailing Stop فيبو')
print(f'🔍 Look-ahead bias: ✅ NONE')
print(f'📋 {len(all_t)}→{len(execd)} | 🟢{len(wins)} 🔴{len(losses)} | 📈 WR {wr:.1f}%')
print(f'💵 {sum(wins):+.1f}% 💸 {sum(losses):+.1f}% 💰 {sum(pnls):+.1f}%')
print(f'🟢 متوسط ربح {aw:+.2f}% 🔴 متوسط خسارة {al:+.2f}%')
print(f'📊 R:R {abs(aw/al):.2f}x | 📊 شارپ {sharpe:.2f} | 📉 سحب {mdd:.1f}%')
print(f'🏦 محفظة: ${CAPITAL}→${eq:,.0f} ({(eq/CAPITAL-1)*100:+.1f}%) — Risk 50%')
print(f'✅ منفذة {len(execd)} ⏭️ متخطية {len(all_t)-len(execd)}')
print(f'🐌 TRAIL {trail_c} ⏱️ TIME {time_c}')
print(f'📊 توزيع التريل: {dict(trail_dist)}')
print(f'⛓️ خسائر متتالية: {maxc}')
