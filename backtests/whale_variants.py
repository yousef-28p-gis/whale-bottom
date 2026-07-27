#!/usr/bin/env python3
"""
WHALE BOTTOM VARIANTS — 15m, 30 days, all parameter combos
Test every variant to find the best configuration
"""
import json, numpy as np, pandas as pd, os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002; INITIAL_CAPITAL = 1000

with open(f'{DATA_DIR}/15m_30d.json') as f:
    all_data = json.load(f)

print(f"🐋 WHALE BOTTOM VARIANTS — {len(all_data)} coins, 15m, 30 days\n")

def simulate(signals, all_data, tp, sl, max_hold):
    signals.sort(key=lambda s: s['date'])
    trades, cap = [], INITIAL_CAPITAL
    active = {}
    for sig in signals:
        coin, ei = sig['coin'], sig['idx']
        if coin in active and active[coin] > ei: continue
        d = all_data[coin]
        c, h, l = np.array(d['close']), np.array(d['high']), np.array(d['low'])
        n = len(c)
        if ei >= n-1: continue
        tp_p = sig['entry']*(1+tp); sl_p = sig['entry']*(1-sl)
        ep = et = ex = None
        for j in range(ei+1, min(ei+max_hold, n)):
            if l[j] <= sl_p: ep=sl_p; et='SL'; ex=j; break
            elif h[j] >= tp_p: ep=tp_p; et='TP'; ex=j; break
        if ep is None:
            end = min(ei+max_hold, n-1); ep=c[end]; et='TIME'; ex=end
        pnl = (ep/sig['entry']-1)*100 - COMMISSION*100
        sz = cap*0.10; cap += sz*pnl/100
        trades.append({'pnl':pnl, 'type':et, 'cap':cap})
        active[coin]=ex
        active={k:v for k,v in active.items() if v>ei}
    return trades, cap

def summarize(name, trades, final_cap):
    if not trades: return f"{name:<45s} 0 trades"
    df = pd.DataFrame(trades)
    wins, losses = df[df['pnl']>0], df[df['pnl']<=0]
    wr = len(wins)/len(df)*100
    eq = np.array([1000]+[t['cap'] for t in trades])
    dd = (eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100
    ret = (final_cap/1000-1)*100
    pf = abs(wins['pnl'].sum()/losses['pnl'].sum()) if len(losses)>0 else 999
    return (f"{name:<45s} {len(df):>4d} | WR {wr:>5.1f}% | Ret {ret:>+6.1f}% | DD {dd.min():>6.2f}% | PF {pf:.2f} | TP:{len(df[df['type']=='TP']):>3d}")

def run_whale(all_data, whale_min, rsi_max, confirm_green, tp, sl, max_hold):
    """Core whale bottom with configurable params"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); v = np.array(data['volume'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        vol_avg = pd.Series(v).rolling(50).mean().values
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(100, n-3):
            if np.isnan(rsi[i]) or vol_avg[i] <= 0: continue
            
            whale = (v[i] - vol_avg[i]) / vol_avg[i]
            if whale < whale_min: continue
            if rsi[i] >= rsi_max: continue
            
            if confirm_green:
                if c[i+1] <= o_arr[i+1]: continue
                entry_idx = i + 2
            else:
                entry_idx = i + 1  # enter immediately
            
            if entry_idx >= n: continue
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# TEST MATRIX
# ═══════════════════════════════════════════════════════

# Part 1: Vary whale threshold (keep RSI<25, green confirm, TP3.5/SL1.5)
print("=" * 95)
print("🔬 PART 1: Whale Threshold (RSI<25, Green✔, TP3.5/SL1.5, MH24h)")
print("=" * 95)
for whale_min in [0.30, 0.40, 0.50, 0.60, 0.75, 1.0]:
    trades, final = run_whale(all_data, whale_min, 25, True, 0.035, 0.015, 96)
    print(f"  {summarize(f'Whale≥{whale_min}  RSI<25  Green✔', trades, final)}")

# Part 2: Vary RSI threshold (keep whale≥0.50, green confirm, TP3.5/SL1.5)
print(f"\n{'='*95}")
print("🔬 PART 2: RSI Threshold (Whale≥0.50, Green✔, TP3.5/SL1.5, MH24h)")
print("=" * 95)
for rsi_max in [20, 25, 30, 35, 40]:
    trades, final = run_whale(all_data, 0.50, rsi_max, True, 0.035, 0.015, 96)
    print(f"  {summarize(f'Whale≥0.50  RSI<{rsi_max}  Green✔', trades, final)}")

# Part 3: Green confirm ON vs OFF
print(f"\n{'='*95}")
print("🔬 PART 3: Green Confirmation (Whale≥0.50, TP3.5/SL1.5, MH24h)")
print("=" * 95)
for rsi_max in [25, 30, 35]:
    for confirm in [True, False]:
        c_label = "✔" if confirm else "✘"
        trades, final = run_whale(all_data, 0.50, rsi_max, confirm, 0.035, 0.015, 96)
        print(f"  {summarize(f'Whale≥0.50  RSI<{rsi_max}  Green{c_label}', trades, final)}")

# Part 4: Best combos with multiple TP/SL
print(f"\n{'='*95}")
print("🔬 PART 4: Best Combos × Multiple TP/SL")
print("=" * 95)

combos = [
    (0.50, 25, True, "Whale≥0.50 RSI<25 Green✔"),
    (0.60, 25, True, "Whale≥0.60 RSI<25 Green✔"),
    (0.50, 20, True, "Whale≥0.50 RSI<20 Green✔"),
    (0.50, 30, True, "Whale≥0.50 RSI<30 Green✔"),
    (0.75, 25, True, "Whale≥0.75 RSI<25 Green✔"),
    (0.50, 25, False, "Whale≥0.50 RSI<25 Green✘"),
    (0.50, 35, True, "Whale≥0.50 RSI<35 Green✔"),
]

tp_sl_combos = [
    (0.035, 0.015, 96, "TP3.5/SL1.5/24h"),
    (0.05, 0.025, 96, "TP5/SL2.5/24h"),
    (0.07, 0.03, 192, "TP7/SL3/48h"),
    (0.05, 0.02, 96, "TP5/SL2/24h"),
    (0.10, 0.04, 192, "TP10/SL4/48h"),
]

for whale_min, rsi_max, confirm, label in combos:
    best_ret = -999
    best_line = ""
    for tp, sl, mh, tl in tp_sl_combos:
        trades, final = run_whale(all_data, whale_min, rsi_max, confirm, tp, sl, mh)
        s = summarize(f"  {label} | {tl}", trades, final)
        print(f"  {s}")
        
        # Track best
        if trades:
            df = pd.DataFrame(trades)
            ret = (final/1000-1)*100
            if ret > best_ret:
                best_ret = ret
                best_line = s

print(f"\n{'='*95}")
print(f"🏆 BEST OVERALL WHALE VARIANT")
print(f"{'='*95}")
print(f"  {best_line}")
print(f"\n✅ All whale variants tested!")
