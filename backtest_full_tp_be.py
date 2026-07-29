#!/usr/bin/env python3
"""Test: Full position TP=1% + Trail SL to BE (no Half TP)"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM = 0.20
CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH = 10; DEV = 1.0; D = DEPTH // 2; CONFIRM = D
MAX_POS = 2; TIME_BARS = 120
TP_PCT = 1.0
SL_PCT = -0.5  # below L2
DIST_FILTER = 0.5

def find_zpatterns(pv):
    pats = []
    for i in range(len(pv)-3):
        p0,p1,p2,p3 = pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1]; B=p2[1]-p1[1]; C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and 0.38<=B/A<=0.79 and p3[1]<p1[1]:
                pats.append((p0,p1,p2,p3))
    return pats

def simulate_one_coin(close, high, low, coin_name):
    n = len(close)
    pivots = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pivots) < 4: return []
    patterns = find_zpatterns(pivots)
    
    trades = []
    for H1,L1,H2,L2 in patterns:
        entry_bar = L2[0] + CONFIRM
        if entry_bar >= n: continue
        entry_price = close[entry_bar]
        dist_pct = (entry_price - L2[1]) / L2[1] * 100
        if dist_pct > DIST_FILTER: continue
        
        sl_init = L2[1] * (1 + SL_PCT/100)  # initial SL below L2
        if sl_init >= entry_price: continue
        
        tp_full = entry_price * (1 + TP_PCT/100)
        be_price = entry_price
        sl_active = sl_init  # current SL (may move to BE)
        
        exit_idx = entry_bar; exit_price = entry_price
        exit_type = 'TIME'; exit_pnl = 0.0
        
        for j in range(entry_bar + 1, min(n, entry_bar + TIME_BARS + 1)):
            bar_high = high[j]; bar_low = low[j]; bar_close = close[j]
            
            # Move SL to BE if price goes above entry
            if bar_high > entry_price:
                sl_active = max(sl_active, be_price)
            
            # Check TP
            if bar_high >= tp_full:
                exit_idx = j; exit_price = tp_full; exit_type = 'TP'
                exit_pnl = (tp_full/entry_price-1)*100 - COMM
                break
            
            # Check SL (close-only)
            if bar_close <= sl_active:
                exit_idx = j; exit_price = bar_close; exit_type = 'SL'
                exit_pnl = (bar_close/entry_price-1)*100 - COMM
                break
        
        else:
            exit_idx = min(entry_bar + TIME_BARS, n-1)
            exit_price = close[exit_idx]
            exit_pnl = (exit_price/entry_price-1)*100 - COMM
        
        trades.append({
            'coin':coin_name,'entry_bar':entry_bar,'exit_bar':exit_idx,
            'entry_price':entry_price,'exit_price':exit_price,
            'exit_type':exit_type,'pnl_net':round(exit_pnl,4),
            'bars':exit_idx-entry_bar
        })
    return trades

# ── Load ──
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

print(f'⏳ {len(COINS)} عملة...', flush=True)
all_trades = []
for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_trades.extend(simulate_one_coin(df['close'].values,df['high'].values,df['low'].values,coin))
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)} — {len(all_trades)}', flush=True)

# ── MAX_POS=2 ──
all_trades.sort(key=lambda t:t['entry_bar'])
executed=[]; active=[]; skipped=0
for t in all_trades:
    active=[a for a in active if a>t['entry_bar']]
    if len(active)>=MAX_POS: skipped+=1; continue
    active.append(t['exit_bar']); executed.append(t)

pnls=[t['pnl_net'] for t in executed]
wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
wr=len(wins)/len(pnls)*100 if pnls else 0
aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0
tp_c=sum(1 for t in executed if t['exit_type']=='TP')
sl_c=sum(1 for t in executed if t['exit_type']=='SL')
time_c=sum(1 for t in executed if t['exit_type']=='TIME')

eq=CAPITAL; peq=CAPITAL; mdd=0; cons=0; maxc=0
for p in pnls:
    eq*=(1+0.10*p/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
    if p<=0: cons+=1; maxc=max(maxc,cons)
    else: cons=0

cs={}; 
for t in executed:
    cn=t['coin']
    if cn not in cs: cs[cn]={'w':0,'l':0,'net':0}
    if t['pnl_net']>0: cs[cn]['w']+=1
    else: cs[cn]['l']+=1
    cs[cn]['net']+=t['pnl_net']
wc=sum(1 for c in cs.values() if c['net']>0)
lc=sum(1 for c in cs.values() if c['net']<=0)

print(f'''
═══ TP=1% كامل + BE (بدون Half TP) ═══
TP=1% | SL=L2-0.5% | BE عند لمس الدخول | TIME=120
Dist<0.5% | MAX_POS=2 | close-only

📋 إشارات: {len(all_trades):,} | ✅ منفذة: {len(executed):,} | ⏭️ متخطية: {skipped:,}
🟢 ربح: {len(wins):,} | 🔴 خسارة: {len(losses):,}
📈 WR: {wr:.1f}%
🟢 متوسط ربح: {aw:+.2f}% | 🔴 متوسط خسارة: {al:+.2f}%
📊 R:R: {aw/abs(al):.1f}x | 📉 سحب: {mdd:.1f}%
🏦 محفظة (10%): ${CAPITAL:,} → ${eq:,.0f} (+{(eq/CAPITAL-1)*100:.1f}%)
🎯 TP: {tp_c} | 🛑 SL: {sl_c} | ⏱️ TIME: {time_c}
🟢 عملات: {wc} | 🔴: {lc}
''')
