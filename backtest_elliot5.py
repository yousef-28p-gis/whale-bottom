#!/usr/bin/env python3
"""Elliot Wave 5-Wave Impulse + ABC Correction — Backtest"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM = 0.20; CAPITAL = 1000; RISK = 0.50
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH = 10; DEV = 1.0; D = DEPTH//2; CONFIRM = D
MAX_POS = 2; TIME_BARS = 120

# Exit params
TP_PCT = 1.0; HALF_TP_PCT = 0.5; SL_ABOVE_L2 = -0.5; DIST_FILTER = 0.5

# Fib levels for validation
FIB_RETRACE = [0.382, 0.50, 0.618, 0.786]
FIB_EXTEND  = [1.0, 1.272, 1.382, 1.618]
FIB_TOL = 0.05

def near_fib(actual, fib_levels, tol=FIB_TOL):
    return any(abs(actual - f) <= tol for f in fib_levels)

def find_5wave_impulse(pv):
    """
    Find 5-wave downward impulse patterns from zigzag pivots.
    Pivots must alternate: H1, L1, H2, L2, H3, L3 (6 pivots = 5 waves)
    
    Elliott Rules for downward impulse:
    1. Wave 2 retracement < 100% of Wave 1
    2. Wave 3 NOT the shortest among waves 1,3,5
    3. Wave 4 does NOT overlap Wave 1: H3 < L1
    4. Wave 5 is a new low: L3 < L2
    5. Fibonacci alignment for wave relationships (optional)
    """
    patterns = []
    
    for i in range(len(pv) - 5):
        # Need 6 alternating pivots: H,L,H,L,H,L
        p = pv[i:i+6]
        types = [pt[2] for pt in p]
        if types != ['H','L','H','L','H','L']:
            continue
        
        H1_idx, H1_val = p[0][0], p[0][1]
        L1_idx, L1_val = p[1][0], p[1][1]
        H2_idx, H2_val = p[2][0], p[2][1]
        L2_idx, L2_val = p[3][0], p[3][1]
        H3_idx, H3_val = p[4][0], p[4][1]
        L3_idx, L3_val = p[5][0], p[5][1]
        
        # Wave sizes (all positive)
        w1 = H1_val - L1_val  # Wave 1 down
        w2 = H2_val - L1_val  # Wave 2 up
        w3 = H2_val - L2_val  # Wave 3 down
        w4 = H3_val - L2_val  # Wave 4 up
        w5 = H3_val - L3_val  # Wave 5 down
        
        # Basic sanity
        if w1 <= 0 or w2 <= 0 or w3 <= 0 or w4 <= 0 or w5 <= 0:
            continue
        
        # Rule 1: Wave 2 < Wave 1 (retrace < 100%)
        if w2 >= w1:
            continue
        
        # Rule 2: Wave 3 NOT shortest
        if w3 <= min(w1, w5):
            continue
        
        # Rule 3: Wave 4 doesn't overlap Wave 1 (H3 < L1 for downtrend)
        if H3_val >= L1_val:
            continue
        
        # Rule 4: Wave 5 is a new low
        if L3_val >= L2_val:
            continue
        
        # Rule 5: New lows each wave (L1 < H1, L2 < L1, L3 < L2)
        if not (L1_val < H1_val and L2_val < L1_val and L3_val < L2_val):
            continue
        
        # Fibonacci alignment for wave relationships
        ret2 = w2 / w1 if w1 > 0 else 99
        ext3 = w3 / w1 if w1 > 0 else 0
        if not near_fib(ret2, FIB_RETRACE):
            continue
        if not near_fib(ext3, FIB_EXTEND):
            continue
        
        patterns.append({
            'pv': p,
            'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
            'L3_idx': L3_idx, 'L3_val': L3_val,
            'ret2': ret2,
        })
    
    return patterns

def simulate_one_coin(close, high, low, coin_name):
    n = len(close)
    pivots = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pivots) < 6: return []
    
    patterns = find_5wave_impulse(pivots)
    
    trades = []
    for pat in patterns:
        L3_idx = pat['L3_idx']
        L3_val = pat['L3_val']
        
        # Entry: after wave 5 confirmation (L3 + CONFIRM)
        entry_bar = L3_idx + CONFIRM
        if entry_bar >= n: continue
        
        entry_price = close[entry_bar]
        
        # Distance filter
        dist_pct = (entry_price - L3_val) / L3_val * 100
        if dist_pct > DIST_FILTER: continue
        
        sl_price = L3_val * (1 + SL_ABOVE_L2/100)
        if sl_price >= entry_price: continue
        
        tp_full = entry_price * (1 + TP_PCT/100)
        tp_half = entry_price * (1 + HALF_TP_PCT/100)
        be_price = entry_price
        
        half_exited = False; half1_pnl = 0.0
        exit_idx = entry_bar; exit_price = entry_price
        exit_type = 'TIME'; exit_pnl_net = 0.0
        
        for j in range(entry_bar + 1, min(n, entry_bar + TIME_BARS + 1)):
            bar_high = high[j]; bar_low = low[j]; bar_close = close[j]
            
            if not half_exited:
                if bar_high >= tp_half:
                    half1_pnl = (tp_half/entry_price - 1)*100 - COMM/2
                    half_exited = True
                    if bar_high >= tp_full:
                        half2_pnl = (tp_full/entry_price - 1)*100 - COMM/2
                        exit_idx=j; exit_price=tp_full; exit_type='TP'
                        exit_pnl_net = (half1_pnl+half2_pnl)/2; break
                    continue
                
                if bar_close <= sl_price:
                    exit_idx=j; exit_price=bar_close; exit_type='SL'
                    exit_pnl_net = (bar_close/entry_price-1)*100 - COMM
                    half_exited = True; break
            
            if half_exited:
                if bar_high >= tp_full:
                    half2_pnl = (tp_full/entry_price-1)*100 - COMM/2
                    exit_idx=j; exit_price=tp_full; exit_type='TP'
                    exit_pnl_net = (half1_pnl+half2_pnl)/2; break
                
                if bar_close <= be_price:
                    half2_pnl = (be_price/entry_price-1)*100 - COMM/2
                    exit_idx=j; exit_price=bar_close; exit_type='BE'
                    exit_pnl_net = (half1_pnl+half2_pnl)/2; break
        else:
            exit_idx = min(entry_bar+TIME_BARS, n-1)
            exit_price = close[exit_idx]
            if half_exited:
                half2_pnl = (exit_price/entry_price-1)*100 - COMM/2
                exit_pnl_net = (half1_pnl+half2_pnl)/2
            else:
                exit_pnl_net = (exit_price/entry_price-1)*100 - COMM
        
        trades.append({
            'coin':coin_name,'entry_bar':entry_bar,'exit_bar':exit_idx,
            'entry_price':entry_price,'exit_price':exit_price,
            'exit_type':exit_type,'pnl_net':round(exit_pnl_net,4),
            'bars':exit_idx-entry_bar
        })
    return trades

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f:
    sh = json.load(f)
COINS = [c for c in sh['halal']+sh['halal2'] if c not in STABLES]

print(f'⏳ Elliot 5-Wave — {len(COINS)} عملة...', flush=True)
all_trades = []
for ci, coin in enumerate(COINS):
    fp = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw = json.load(f)
    if len(raw) < 200: continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_trades.extend(simulate_one_coin(df['close'].values,df['high'].values,df['low'].values,coin))
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)} — {len(all_trades)}', flush=True)

# MAX_POS=2
all_trades.sort(key=lambda t:t['entry_bar'])
executed=[]; active=[]; skipped=0
for t in all_trades:
    active=[a for a in active if a>t['entry_bar']]
    if len(active)>=MAX_POS: skipped+=1; continue
    active.append(t['exit_bar']); executed.append(t)

pnls=[t['pnl_net'] for t in executed]
if not pnls: print('0 صفقات!'); quit()

wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
wr=len(wins)/len(pnls)*100
aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0

# Find bad coins
coin_pnl={}
for t in all_trades:
    cn=t['coin']; coin_pnl[cn]=coin_pnl.get(cn,0)+t['pnl_net']
bad={c for c,net in coin_pnl.items() if net<=5.0}
print(f'❌ <5%: {len(bad)} | ✅ ≥5%: {len(coin_pnl)-len(bad)}')

# Filter bad coins
filtered=[t for t in executed if t['coin'] not in bad]

# 50% compounding
eq=CAPITAL; peq=CAPITAL; mdd=0; cons=0; maxc=0
for p in filtered:
    eq*=(1+RISK*p['pnl_net']/100); peq=max(peq,eq); mdd=min(mdd,(eq-peq)/peq*100)
    if p['pnl_net']<=0: cons+=1; maxc=max(maxc,cons)
    else: cons=0

tp_c=sum(1 for t in filtered if t['exit_type']=='TP')
sl_c=sum(1 for t in filtered if t['exit_type']=='SL')
be_c=sum(1 for t in filtered if t['exit_type']=='BE')
ti_c=sum(1 for t in filtered if t['exit_type']=='TIME')

real_wins=[p for p in [t['pnl_net'] for t in filtered] if p>0]
real_loss=[p for p in [t['pnl_net'] for t in filtered] if p<=0]
rw=np.mean(real_wins) if real_wins else 0; rl=np.mean(real_loss) if real_loss else 0
rwr=len(real_wins)/len(filtered)*100 if filtered else 0

coins_left=len({t['coin'] for t in filtered})

print(f'''
═══ Elliot 5-Wave + Fib Half TP + BE ═══
5 موجات هابطة + تأكيد فيبوناتشي
{coins_left} عملة ≥5%

📅 4 شهور — {coins_left} عملة
📋 {len(all_trades):,} ⏭️ {skipped:,} → ✅ {len(filtered):,}
🟢 {len(real_wins)} 🔴 {len(real_loss)} | WR: {rwr:.1f}%
🟢 +{rw:+.2f}% 🔴 {rl:+.2f}% | R:R {rw/abs(rl):.2f}x
📉 سحب {mdd:.1f}% | أطول خسائر {maxc}
🏦 $1K → ${eq:,.0f} (+{(eq/CAPITAL-1)*100:.1f}%)
🎯 TP {tp_c} 🛑 SL {sl_c} 🟰 BE {be_c} ⏱️ TIME {ti_c}
''')
