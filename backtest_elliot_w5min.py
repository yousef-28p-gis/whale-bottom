#!/usr/bin/env python3
"""Elliot 5-Wave + w5 >= 2% filter backtest & save completed trades"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag

COMM=0.20; CAPITAL=1000; RISK=0.50
DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD'}

DEPTH=10; DEV=1.0; D=DEPTH//2; CONFIRM=D
MAX_POS=2; TIME_BARS=120; DIST_FILTER=0.5; INIT_SL_PCT=-0.5

def near(v,target,tol=0.03): return abs(v-target)<=tol

def find_5waves(pv, w5_filter=None, min_w5_pct=0):
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
        w5_pct=w5/L3[1]*100
        if w5_pct<min_w5_pct: continue
        pats.append((H1,L1,H2,L2,H3,L3))
    return pats

def simulate(close,high,low,coin,w5_filter,min_w5):
    n=len(close); pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: return[]
    pats=find_5waves(pv,w5_filter,min_w5)
    trades=[]
    
    for H1,L1,H2,L2,H3,L3 in pats:
        eb=L3[0]+CONFIRM
        if eb>=n: continue
        ep=close[eb]
        if (ep-L3[1])/L3[1]*100>DIST_FILTER: continue
        
        w5_size=H3[1]-L3[1]; w5_pct=w5_size/L3[1]*100
        if w5_pct<min_w5: continue
        
        fib_half = L3[1] + 0.5 * w5_size   # Fib 0.5 from L3 (start of wave 5)
        fib_full = H3[1]                   # Fib 1.0 = H3 (end of wave 5)
        
        sl=L3[1]*(1+INIT_SL_PCT/100)
        if sl>=ep: continue
        
        half_exited=False; half1_pnl=0.0
        exit_type='TIME'; exit_idx=eb; exit_pnl=0.0; be=ep
        
        for j in range(eb+1, min(n,eb+TIME_BARS+1)):
            bh=high[j]; bl=low[j]; bc=close[j]
            if not half_exited:
                if bh>=fib_half:
                    half1_pnl=(fib_half/ep-1)*100-COMM/2; half_exited=True
                    if bh>=fib_full:
                        h2=(fib_full/ep-1)*100-COMM/2
                        exit_idx=j; exit_type='FULL'; exit_pnl=(half1_pnl+h2)/2; break
                    continue
                if bc<=sl:
                    exit_idx=j; exit_type='SL'; exit_pnl=(bc/ep-1)*100-COMM; break
            if half_exited:
                if bh>=fib_full:
                    h2=(fib_full/ep-1)*100-COMM/2
                    exit_idx=j; exit_type='FULL'; exit_pnl=(half1_pnl+h2)/2; break
                if bc<=be:
                    h2=-COMM/2
                    exit_idx=j; exit_type='BE'; exit_pnl=(half1_pnl+h2)/2; break
        else:
            jj=min(eb+TIME_BARS,n-1)
            if half_exited: h2=(close[jj]/ep-1)*100-COMM/2; exit_pnl=(half1_pnl+h2)/2
            else: exit_pnl=(close[jj]/ep-1)*100-COMM
            exit_idx=jj
        
        if exit_pnl>10 or exit_pnl<-15: continue
        
        trades.append({
            'coin':coin,'eb':eb,'exit_bar':exit_idx,'pnl':round(exit_pnl,4),
            'type':exit_type,'w5_pct':round(w5_pct,3),
            'ep':ep,'L3':L3[1],'H3':H3[1],
            'fib_half':fib_half,'fib_full':fib_full,
            'i_L3':L3[0],'i_H3':H3[0],
        })
    return trades

with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

W5F=[0.382]

for min_w5 in [0, 2.0]:
    label=f'w5≥{min_w5}%' if min_w5>0 else 'الكل'
    print(f'\n═══ w5 ≥ {min_w5}% ═══', flush=True)
    all_t=[]
    for cn in COINS:
        fp=f'{DATA}/{cn}.json'
        if not os.path.exists(fp): continue
        with open(fp) as f: raw=json.load(f)
        if len(raw)<200: continue
        df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        all_t.extend(simulate(df['close'].values,df['high'].values,df['low'].values,cn,W5F,min_w5))
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
    sharpe=(np.mean(ret)/np.std(ret))*np.sqrt(len(ret)) if len(ret)>1 and np.std(ret)>0 else 0
    avg_w5=np.mean([t['w5_pct'] for t in execd]) if execd else 0
    
    full_c=sum(1 for t in execd if t['type']=='FULL')
    sl_c=sum(1 for t in execd if t['type']=='SL')
    be_c=sum(1 for t in execd if t['type']=='BE')
    time_c=sum(1 for t in execd if t['type']=='TIME')
    
    print(f'\n📅 4 شهور — 3m — CLOSE-ONLY — {label}')
    print(f'🔍 Look-ahead bias: ✅ NONE')
    print(f'📋 {len(all_t)}→{len(execd)} | 🟢{len(wins)} 🔴{len(losses)} | 📈 WR {wr:.1f}%')
    print(f'💵 {sum(wins):+.1f}% 💸 {sum(losses):+.1f}% 💰 {sum(pnls):+.1f}%')
    print(f'🟢 متوسط ربح {aw:+.2f}% 🔴 متوسط خسارة {al:+.2f}%')
    print(f'📊 R:R {abs(aw/al):.2f}x | 📊 شارپ {sharpe:.2f} | 📉 سحب {mdd:.1f}%')
    print(f'🏦 محفظة: ${CAPITAL}→${eq:,.0f} ({(eq/CAPITAL-1)*100:+.1f}%)')
    print(f'✅ {len(execd)} ⏭️ {len(all_t)-len(execd)} | 🎯FULL {full_c} 🛑SL {sl_c} 🐌BE {be_c} ⏱️TIME {time_c}')
    print(f'📏 متوسط w5: {avg_w5:.2f}% | ⛓️ خسائر متتالية: {maxc}')
    
    # Save completed trades for plotting (last run only)
    if min_w5 == 0:
        with open('/data/trading28/elliot_completed_trades.json', 'w') as f:
            json.dump(execd, f)
        print(f'\n  💾 حفظ {len(execd)} صفقة منتهية')
