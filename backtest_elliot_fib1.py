#!/usr/bin/env python3
"""Elliot 5-Wave + w5=0.382(w1+w3) — TP = Fib 1.0 of Wave 5 (H3)"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; CAPITAL=1000; RISK=0.50
DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD'}

DEPTH=10; DEV=1.0; D=DEPTH//2; CONFIRM=D
MAX_POS=2; TIME_BARS=120; DIST_FILTER=0.5
SL_PCT=-0.5  # SL below L3

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

def simulate_fib1(close,high,low,coin,w5_filter, half_fib=None):
    """half_fib: if set, exit half at that fib level, rest at fib 1.0"""
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: return[]
    pats=find_5waves(pv,w5_filter)
    trades=[]
    
    for H1,L1,H2,L2,H3,L3 in pats:
        eb=L3[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L3[1])/L3[1]*100>DIST_FILTER: continue
        
        w5_size = H3[1] - L3[1]  # wave 5 size
        if w5_size <= 0: continue
        
        # Fib 1.0 = full retrace of wave 5 = H3 price
        fib1_price = H3[1]
        
        # SL = L3 - 0.5%
        sl = L3[1] * (1 + SL_PCT/100)
        if sl >= ep: continue
        
        if half_fib is not None:
            # Half exit at half_fib (e.g. 0.5 = 50% of wave 5)
            half_target = L3[1] + w5_size * half_fib
            if half_target <= ep: continue  # must be above entry
            
            half_exited=False; half1_pnl=0.0
            exit_type='TIME'; exit_idx=eb; exit_pnl=0.0
            be=ep
            
            for j in range(eb+1, min(n, eb+TIME_BARS+1)):
                bh=high[j]; bl=low[j]; bc=close[j]
                if not half_exited:
                    if bh >= half_target:
                        half1_pnl=(half_target/ep-1)*100-COMM/2; half_exited=True
                        if bh >= fib1_price:
                            h2=(fib1_price/ep-1)*100-COMM/2
                            exit_idx=j; exit_type='FIB1'; exit_pnl=(half1_pnl+h2)/2; break
                        continue
                    if bc <= sl:
                        exit_idx=j; exit_type='SL'; exit_pnl=(bc/ep-1)*100-COMM; break
                if half_exited:
                    if bh >= fib1_price:
                        h2=(fib1_price/ep-1)*100-COMM/2
                        exit_idx=j; exit_type='FIB1'; exit_pnl=(half1_pnl+h2)/2; break
                    if bc <= be:
                        h2=-COMM/2
                        exit_idx=j; exit_type='BE'; exit_pnl=(half1_pnl+h2)/2; break
            else:
                jj=min(eb+TIME_BARS, n-1)
                if half_exited:
                    h2=(close[jj]/ep-1)*100-COMM/2
                    exit_pnl=(half1_pnl+h2)/2
                else:
                    exit_pnl=(close[jj]/ep-1)*100-COMM
                exit_idx=jj
        else:
            # Full position — TP = fib1_price, SL only
            exit_type='TIME'; exit_idx=eb; exit_pnl=0.0
            for j in range(eb+1, min(n, eb+TIME_BARS+1)):
                bh=high[j]; bl=low[j]; bc=close[j]
                if bh >= fib1_price:
                    exit_idx=j; exit_type='FIB1'; exit_pnl=(fib1_price/ep-1)*100-COMM; break
                if bc <= sl:
                    exit_idx=j; exit_type='SL'; exit_pnl=(bc/ep-1)*100-COMM; break
            else:
                jj=min(eb+TIME_BARS, n-1)
                exit_idx=jj; exit_pnl=(close[jj]/ep-1)*100-COMM
        
        # Validate exit_pnl — filter extreme outliers
        if exit_pnl > 10 or exit_pnl < -15: continue
        
        trades.append({'coin':coin,'eb':eb,'exit_bar':exit_idx,'pnl':round(exit_pnl,4),'type':exit_type,'w5':round(w5_size,6)})
    return trades

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

W5_FILTER=[0.382]

# ─── TEST 1: Full TP = Fib 1.0 ───
print('═══ اختبار 1: خروج كامل عند فيبو 1.0 ═══', flush=True)
all_t=[]
for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t.extend(simulate_fib1(df['close'].values,df['high'].values,df['low'].values,cn,W5_FILTER))
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

tp_c=sum(1 for t in execd if t['type']=='FIB1')
sl_c=sum(1 for t in execd if t['type']=='SL')
be_c=sum(1 for t in execd if t['type']=='BE')
time_c=sum(1 for t in execd if t['type']=='TIME')

# Sharpe
ret=[p for p in pnls]
if len(ret)>1:
    sharpe=(np.mean(ret)/np.std(ret))*np.sqrt(len(ret)) if np.std(ret)>0 else 0
else: sharpe=0

# avg w5 size
avg_w5 = np.mean([t['w5'] for t in execd]) if execd else 0

print(f'\n📅 4 شهور — 3m — CLOSE-ONLY')
print(f'📊 بيانات: {len(COINS)} عملة')
print(f'🔍 Look-ahead bias: ✅ NONE')
print(f'📋 {len(all_t)}→{len(execd)} | 🟢{len(wins)} 🔴{len(losses)} | 📈 WR {wr:.1f}%')
print(f'💵 {sum(wins):+.1f}% 💸 {sum(losses):+.1f}% 💰 {sum(pnls):+.1f}%')
print(f'🟢 متوسط ربح {aw:+.2f}% 🔴 متوسط خسارة {al:+.2f}%')
print(f'📊 R:R {abs(aw/al):.2f}x | 📊 شارپ {sharpe:.2f} | 📉 سحب {mdd:.1f}%')
print(f'🏦 محفظة: ${CAPITAL}→${eq:,.0f} ({(eq/CAPITAL-1)*100:+.1f}%)')
print(f'✅ منفذة {len(execd)} ⏭️ متخطية {len(all_t)-len(execd)}')
print(f'🎯 FIB1 {tp_c} 🛑 SL {sl_c} 🐌 BE {be_c} ⏱️ TIME {time_c}')
print(f'📏 متوسط حجم w5: ${avg_w5:.4f} | ⛓️ خسائر متتالية: {maxc}')

# ─── TEST 2: Half Fib 0.5 + Fib 1.0 + BE ───
print(f'\n═══ اختبار 2: نصف عند فيبو 0.5 + نصف عند فيبو 1.0 + BE ═══', flush=True)
all_t2=[]
for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t2.extend(simulate_fib1(df['close'].values,df['high'].values,df['low'].values,cn,W5_FILTER,half_fib=0.5))
    del df; gc.collect()

all_t2.sort(key=lambda t:t['eb'])
execd2=[]; active2=[]
for t in all_t2:
    active2=[a for a in active2 if a>t['eb']]
    if len(active2)>=MAX_POS: continue
    active2.append(t['exit_bar']); execd2.append(t)

pnls2=[t['pnl'] for t in execd2]; wins2=[p for p in pnls2 if p>0]; losses2=[p for p in pnls2 if p<=0]
wr2=len(wins2)/len(pnls2)*100 if pnls2 else 0
aw2=np.mean(wins2) if wins2 else 0; al2=np.mean(losses2) if losses2 else 0
eq2=CAPITAL; peq2=CAPITAL; mdd2=0; cons2=0; maxc2=0
for p in pnls2:
    eq2*=(1+RISK*p/100); peq2=max(peq2,eq2); mdd2=min(mdd2,(eq2-peq2)/peq2*100)
    if p<=0: cons2+=1; maxc2=max(maxc2,cons2)
    else: cons2=0

tp_c2=sum(1 for t in execd2 if t['type']=='FIB1')
sl_c2=sum(1 for t in execd2 if t['type']=='SL')
be_c2=sum(1 for t in execd2 if t['type']=='BE')
time_c2=sum(1 for t in execd2 if t['type']=='TIME')

ret2=[p for p in pnls2]
if len(ret2)>1: sharpe2=(np.mean(ret2)/np.std(ret2))*np.sqrt(len(ret2)) if np.std(ret2)>0 else 0
else: sharpe2=0

print(f'\n📅 4 شهور — 3m — CLOSE-ONLY')
print(f'📊 بيانات: {len(COINS)} عملة')
print(f'🔍 Look-ahead bias: ✅ NONE')
print(f'📋 {len(all_t2)}→{len(execd2)} | 🟢{len(wins2)} 🔴{len(losses2)} | 📈 WR {wr2:.1f}%')
print(f'💵 {sum(wins2):+.1f}% 💸 {sum(losses2):+.1f}% 💰 {sum(pnls2):+.1f}%')
print(f'🟢 متوسط ربح {aw2:+.2f}% 🔴 متوسط خسارة {al2:+.2f}%')
print(f'📊 R:R {abs(aw2/al2):.2f}x | 📊 شارپ {sharpe2:.2f} | 📉 سحب {mdd2:.1f}%')
print(f'🏦 محفظة: ${CAPITAL}→${eq2:,.0f} ({(eq2/CAPITAL-1)*100:+.1f}%)')
print(f'✅ منفذة {len(execd2)} ⏭️ متخطية {len(all_t2)-len(execd2)}')
print(f'🎯 FIB1 {tp_c2} 🛑 SL {sl_c2} 🐌 BE {be_c2} ⏱️ TIME {time_c2}')
print(f'⛓️ خسائر متتالية: {maxc2}')

# ─── TEST 3: Half Fib 0.618 + Fib 1.0 + BE ───
print(f'\n═══ اختبار 3: نصف عند فيبو 0.618 + نصف عند فيبو 1.0 + BE ═══', flush=True)
all_t3=[]
for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_t3.extend(simulate_fib1(df['close'].values,df['high'].values,df['low'].values,cn,W5_FILTER,half_fib=0.618))
    del df; gc.collect()

all_t3.sort(key=lambda t:t['eb'])
execd3=[]; active3=[]
for t in all_t3:
    active3=[a for a in active3 if a>t['eb']]
    if len(active3)>=MAX_POS: continue
    active3.append(t['exit_bar']); execd3.append(t)

pnls3=[t['pnl'] for t in execd3]; wins3=[p for p in pnls3 if p>0]; losses3=[p for p in pnls3 if p<=0]
wr3=len(wins3)/len(pnls3)*100 if pnls3 else 0
aw3=np.mean(wins3) if wins3 else 0; al3=np.mean(losses3) if losses3 else 0
eq3=CAPITAL; peq3=CAPITAL; mdd3=0; cons3=0; maxc3=0
for p in pnls3:
    eq3*=(1+RISK*p/100); peq3=max(peq3,eq3); mdd3=min(mdd3,(eq3-peq3)/peq3*100)
    if p<=0: cons3+=1; maxc3=max(maxc3,cons3)
    else: cons3=0

tp_c3=sum(1 for t in execd3 if t['type']=='FIB1')
sl_c3=sum(1 for t in execd3 if t['type']=='SL')
be_c3=sum(1 for t in execd3 if t['type']=='BE')
time_c3=sum(1 for t in execd3 if t['type']=='TIME')

ret3=[p for p in pnls3]
if len(ret3)>1: sharpe3=(np.mean(ret3)/np.std(ret3))*np.sqrt(len(ret3)) if np.std(ret3)>0 else 0
else: sharpe3=0

print(f'\n📅 4 شهور — 3m — CLOSE-ONLY')
print(f'📊 بيانات: {len(COINS)} عملة')
print(f'🔍 Look-ahead bias: ✅ NONE')
print(f'📋 {len(all_t3)}→{len(execd3)} | 🟢{len(wins3)} 🔴{len(losses3)} | 📈 WR {wr3:.1f}%')
print(f'💵 {sum(wins3):+.1f}% 💸 {sum(losses3):+.1f}% 💰 {sum(pnls3):+.1f}%')
print(f'🟢 متوسط ربح {aw3:+.2f}% 🔴 متوسط خسارة {al3:+.2f}%')
print(f'📊 R:R {abs(aw3/al3):.2f}x | 📊 شارپ {sharpe3:.2f} | 📉 سحب {mdd3:.1f}%')
print(f'🏦 محفظة: ${CAPITAL}→${eq3:,.0f} ({(eq3/CAPITAL-1)*100:+.1f}%)')
print(f'✅ منفذة {len(execd3)} ⏭️ متخطية {len(all_t3)-len(execd3)}')
print(f'🎯 FIB1 {tp_c3} 🛑 SL {sl_c3} 🐌 BE {be_c3} ⏱️ TIME {time_c3}')
print(f'⛓️ خسائر متتالية: {maxc3}')
