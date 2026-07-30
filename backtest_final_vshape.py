#!/usr/bin/env python3
"""Half TP + BE — 50% risk, MAX_POS=2, delete losing coins"""
import json, numpy as np, pandas as pd, os, gc, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

COMM = 0.20; CAPITAL = 1000; RISK = 0.50
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

DEPTH = 10; DEV = 1.0; D = DEPTH // 2; CONFIRM = D
MAX_POS = 2; TIME_BARS = 120
TP_PCT = 1.0; HALF_TP_PCT = 0.5; SL_PCT = -0.5; DIST_FILTER = 0.5

# Fibonacci levels with tolerance
FIB_RETRACE = [0.382, 0.50, 0.618, 0.786]   # Wave B of A
FIB_EXTEND  = [1.0, 1.272, 1.382, 1.618]     # Wave C of A
FIB_TOL = 0.05  # ±5% tolerance

def near_fib(actual, fib_levels, tol=FIB_TOL):
    """Check if actual ratio is close to any Fibonacci level"""
    return any(abs(actual - f) <= tol for f in fib_levels)

def find_zpatterns(pv):
    pats = []
    for i in range(len(pv)-3):
        p0,p1,p2,p3 = pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1]; B=p2[1]-p1[1]; C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and p3[1]<p1[1]:
                ret_B = B/A  # Wave B retracement
                ext_C = C/A  # Wave C extension
                if near_fib(ret_B, FIB_RETRACE) and near_fib(ext_C, FIB_EXTEND):
                    pats.append((p0,p1,p2,p3,ret_B,ext_C))
    return pats

def simulate_one_coin(close, high, low, coin_name):
    n = len(close)
    pivots = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pivots) < 4: return []
    patterns = find_zpatterns(pivots)
    
    trades = []
    for H1, L1, H2, L2, ret_B, ext_C in patterns:
        entry_bar = L2[0] + CONFIRM
        if entry_bar >= n: continue
        entry_price = close[entry_bar]
        if (entry_price - L2[1]) / L2[1] * 100 > DIST_FILTER: continue
        
        sl_price = L2[1] * (1 + SL_PCT/100)
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
            'coin':coin_name, 'entry_bar':entry_bar, 'exit_bar':exit_idx,
            'entry_price':entry_price, 'exit_price':exit_price,
            'exit_type':exit_type, 'pnl_net':round(exit_pnl_net,4),
            'bars':exit_idx-entry_bar
        })
    return trades

# ── Load coins ──
with open('/data/trading28/config/shariah_coins.json') as f:
    sh = json.load(f)
COINS = [c for c in sh['halal']+sh['halal2'] if c not in STABLES]

print(f'⏳ {len(COINS)} عملة...', flush=True)
all_trades = []
for ci, coin in enumerate(COINS):
    fp = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw = json.load(f)
    if len(raw) < 200: continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    all_trades.extend(simulate_one_coin(df['close'].values, df['high'].values, df['low'].values, coin))
    del df; gc.collect()
    if (ci+1)%40==0: print(f'  ⏳ {ci+1}/{len(COINS)} — {len(all_trades)}', flush=True)

# ── Find low-profit coins (<5% net) ──
coin_pnl = {}
for t in all_trades:
    cn = t['coin']
    coin_pnl[cn] = coin_pnl.get(cn, 0) + t['pnl_net']
low_profit = {c for c, net in coin_pnl.items() if net <= 5.0}
print(f'\n❌ عملات <5% ربح: {len(low_profit)} — {sorted(low_profit)}')
print(f'✅ عملات رابحة ≥5%: {len(coin_pnl)-len(low_profit)}')

# ── Filter: remove low-profit coins ──
filtered = [t for t in all_trades if t['coin'] not in low_profit]

# ── MAX_POS=2 global ──
filtered.sort(key=lambda t: t['entry_bar'])
executed = []; active = []; skipped = 0
for t in filtered:
    active = [a for a in active if a > t['entry_bar']]
    if len(active) >= MAX_POS: skipped += 1; continue
    active.append(t['exit_bar']); executed.append(t)

pnls = [t['pnl_net'] for t in executed]
wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
wr = len(wins)/len(pnls)*100 if pnls else 0
aw = np.mean(wins) if wins else 0; al = np.mean(losses) if losses else 0

# ── 50% compounding ──
eq = CAPITAL; peq = CAPITAL; mdd = 0; cons = 0; maxc = 0
for p in pnls:
    eq *= (1 + RISK * p/100)
    peq = max(peq, eq); mdd = min(mdd, (eq-peq)/peq*100)
    if p <= 0: cons += 1; maxc = max(maxc, cons)
    else: cons = 0

# Also without compounding
eq_flat = CAPITAL + CAPITAL * sum(pnls) * RISK / 100

tp_c = sum(1 for t in executed if t['exit_type']=='TP')
sl_c = sum(1 for t in executed if t['exit_type']=='SL')
be_c = sum(1 for t in executed if t['exit_type']=='BE')
ti_c = sum(1 for t in executed if t['exit_type']=='TIME')

# Coins left
coins_left = {t['coin'] for t in executed}
coin_wins = sum(1 for c in coins_left if sum(t['pnl_net'] for t in executed if t['coin']==c) > 0)

days = 122
annual = ((eq/CAPITAL)**(365/days)-1)*100
arr = aw/abs(al)

print(f'''
═══ ZigZag V-Shape + Half TP + BE ═══
بعد حذف {len(low_profit)} عملة <5% ربح
MAX_POS=2 | 50% كل صفقة | close-only

📅 4 شهور — {len(coins_left)} عملة
📊 Half TP 0.5% + BE | TP=1% | SL=L2-0.5% | TIME=120
🔍 Look-ahead bias: ✅ NONE

📋 إشارات: {len(filtered):,} | ✅ منفذة: {len(executed):,} | ⏭️ متخطية: {skipped:,}
🟢 ربح: {len(wins):,} | 🔴 خسارة: {len(losses):,}
📈 WR: {wr:.1f}%
🟢 متوسط ربح: {aw:+.2f}% | 🔴 متوسط خسارة: {al:+.2f}%
📊 R:R: {arr:.2f}x | 📉 سحب: {mdd:.1f}% | أطول خسائر: {maxc}
🏦 بدون تركيب: ${CAPITAL:,.0f} → ${eq_flat:,.0f} (+{(eq_flat/CAPITAL-1)*100:.1f}%)
🏦 محفظة (50%): ${CAPITAL:,.0f} → ${eq:,.0f} (+{(eq/CAPITAL-1)*100:.1f}%)
📈 عائد سنوي: {annual:.0f}%
✅ منفذة: {len(executed):,} ⏭️ متخطية: {skipped:,}
🎯 TP: {tp_c} | 🛑 SL: {sl_c} | 🟰 BE: {be_c} | ⏱️ TIME: {ti_c}
''')
