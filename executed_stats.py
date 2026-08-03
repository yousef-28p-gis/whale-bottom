#!/usr/bin/env python3
"""Recompute portfolio stats for EXECUTED trades only"""
import json, numpy as np, os, time, gc, pandas as pd

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

TP=1.3; SL=0.5; PL=12; TRAIL=0.02; MAX_H=4; WHALE_MIN=0.10; RSI_MAX=35

def compute_indicators(df):
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vm!=0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta>0, delta, 0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta<0, -delta, 0)).rolling(14).mean().values
    df['rsi'] = 100 - 100/(1 + gain/np.where(loss!=0, loss, np.nan))
    return df

def find_signals(df):
    n = len(df)
    if n < 100: return np.array([], dtype=int)
    whale = df['whale'].values; spike = df['spike'].values; rsi = df['rsi'].values
    mask = (whale >= WHALE_MIN) & (spike >= 1.5) & (rsi < RSI_MAX) & ~np.isnan(whale) & ~np.isnan(spike) & ~np.isnan(rsi)
    mask[:50] = False
    has_prev = np.zeros(n, dtype=bool)
    for shift in [1, 2, 3]:
        s = np.zeros(n, dtype=bool); s[shift:] = mask[:-shift]; has_prev |= s
    mask &= ~has_prev
    next_green = np.zeros(n, dtype=bool)
    next_green[:-1] = df['close'].values[1:] > df['open'].values[1:]
    mask &= next_green
    return np.where(mask)[0]

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
EXCLUDE = {'ETH','BTC','TRX','XRP','QI','LSK','GLMR','XTZ','YFI',
           'TLM','0G','LA','DYM','VANRY','SENT','VET','COOKIE','HEI','ACT','CKB','AR','RSR','AXS','XEC'}
COINS = [c for c in COINS if c not in EXCLUDE]

print(f"⏳ جمع الصفقات ({len(COINS)} عملة)...", flush=True)
all_trades = []
processed = 0; t0 = time.time()

for coin in COINS:
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    if len(raw) < 200: del raw; continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw

    df = compute_indicators(df)
    close_arr = df['close'].values
    ts_arr = df['ts'].values.astype('datetime64[ns]').astype('int64')

    idxs = find_signals(df)
    if len(idxs) == 0: continue

    max_bars = int(MAX_H*60/TF_MIN)
    tp_r = 1+TP/100; sl_r = 1-SL/100; tr_r = 1-TRAIL/100
    active = []; sig_map = dict(zip(idxs, close_arr[idxs]))

    for i in range(len(df)):
        cur = close_arr[i]
        if i in sig_map:
            active.append({'symbol':coin,'entry':sig_map[i],
                'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                'pl_ok':False,'peak':sig_map[i],'trail':sig_map[i],
                'entry_i':i,'entry_ns':int(ts_arr[i])})
        for j in range(len(active)-1,-1,-1):
            p = active[j]; e = p['entry']; bh = i-p['entry_i']
            if bh>=max_bars:
                p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TIME'; p['exit_ns']=int(ts_arr[i]); all_trades.append(p); del active[j]
            elif cur>=p['tp']:
                p['pnl']=round(TP-COMM,4); p['exit_type']='TP'; p['exit_ns']=int(ts_arr[i]); all_trades.append(p); del active[j]
            elif cur<=p['sl']:
                p['pnl']=round(-SL-COMM,4); p['exit_type']='SL'; p['exit_ns']=int(ts_arr[i]); all_trades.append(p); del active[j]
            elif p['pl_ok']:
                if cur>p['peak']: p['peak']=cur; p['trail']=cur*tr_r
                if cur<=p['trail']:
                    p['pnl']=round((p['trail']/e-1)*100-COMM,4); p['exit_type']='TRAIL'; p['exit_ns']=int(ts_arr[i]); all_trades.append(p); del active[j]
            else:
                pl_p = e+(p['tp']-e)*(PL/100)
                if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*tr_r

    del df; gc.collect()
    processed += 1
    if processed % 50 == 0:
        print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"✅ {time.time()-t0:.0f}s | {len(all_trades):,} إشارة\n", flush=True)

# Sort by entry time
all_trades.sort(key=lambda t: t['entry_ns'])

# Portfolio simulation tracking executed trades
equity = float(CAPITAL)
peak = float(CAPITAL)
max_dd = 0.0
active_slots = [None, None]
executed = 0; skipped = 0
executed_trades = []

for t in all_trades:
    entry_ns = t['entry_ns']
    exit_ns = t['exit_ns']
    pnl_pct = t['pnl']

    # Free completed slots
    for s in range(MAX_POS):
        if active_slots[s] is not None:
            slot_exit_ns, slot_pnl = active_slots[s]
            if slot_exit_ns <= entry_ns:
                pos_cap = equity * 0.5
                pnl_dollar = pos_cap * (slot_pnl / 100)
                equity += pnl_dollar
                active_slots[s] = None
                if equity > peak: peak = equity
                dd = (equity - peak) / peak * 100
                if dd < max_dd: max_dd = dd

    free = -1
    for s in range(MAX_POS):
        if active_slots[s] is None:
            free = s; break

    if free == -1:
        skipped += 1
        continue

    executed += 1
    active_slots[free] = (exit_ns, pnl_pct)
    executed_trades.append(t)

# Close remaining
for s in range(MAX_POS):
    if active_slots[s] is not None:
        slot_exit_ns, slot_pnl = active_slots[s]
        pos_cap = equity * 0.5
        pnl_dollar = pos_cap * (slot_pnl / 100)
        equity += pnl_dollar
        active_slots[s] = None

if equity > peak: peak = equity
dd = (equity - peak) / peak * 100
if dd < max_dd: max_dd = dd

# Stats for executed trades only
et = executed_trades
wins = sum(1 for t in et if t['pnl'] > 0)
losses = sum(1 for t in et if t['pnl'] <= 0)
wr = wins/len(et)*100 if et else 0
total_profit = sum(t['pnl'] for t in et if t['pnl'] > 0)
total_loss = sum(t['pnl'] for t in et if t['pnl'] <= 0)
avg_win = np.mean([t['pnl'] for t in et if t['pnl'] > 0]) if wins else 0
avg_loss = np.mean([t['pnl'] for t in et if t['pnl'] <= 0]) if losses else 0
rr = avg_win/abs(avg_loss) if avg_loss != 0 else 0
tp_c = sum(1 for t in et if t['exit_type'] == 'TP')
sl_c = sum(1 for t in et if t['exit_type'] == 'SL')
tr_c = sum(1 for t in et if t['exit_type'] == 'TRAIL')
tm_c = sum(1 for t in et if t['exit_type'] == 'TIME')

returns = [t['pnl'] for t in et]
sharpe = np.mean(returns)/np.std(returns)*np.sqrt(len(returns)) if len(returns)>1 else 0

# Without compounding
fixed_pnl = sum(t['pnl'] for t in et)
fixed_equity = 1000 + fixed_pnl * 5  # $5 per trade (0.5% of $1000)

print(f"{'='*70}")
print(f"📊 المنفذة فقط — 188 عملة — MAX_POS=2 عالمي")
print(f"{'='*70}")
print(f"📋 إجمالي الإشارات: {len(all_trades):,}")
print(f"✅ منفذة: {executed:,} | ⏭️ متخطية: {skipped:,} ({skipped/len(all_trades)*100:.0f}%)")
print(f"🟢 ربح: {wins:,} | 🔴 خسارة: {losses:,}")
print(f"📈 WR: {wr:.1f}%")
print(f"💵 إجمالي الربح: +{total_profit:.1f}% | 💸 إجمالي الخسارة: {total_loss:.1f}%")
print(f"💰 صافي (مجموع): {total_profit+total_loss:+.1f}%")
print(f"🟢 متوسط ربح: +{avg_win:.2f}% | 🔴 متوسط خسارة: {avg_loss:.2f}%")
print(f"📊 R:R: {rr:.2f}x | 📊 شارپ: {sharpe:.2f}")
print(f"🏦 محفظة (50% تركيب): $1,000 → ${equity:,.0f}")
print(f"🏦 بدون تركيب ($5/صفقة): $1,000 → ${fixed_equity:,.0f}")
print(f"📉 أقصى سحب: {max_dd:.1f}%")
print(f"🎯 TP:{tp_c:,} 🛑 SL:{sl_c:,} 🐌 TRAIL:{tr_c:,} ⏱️ TIME:{tm_c:,}")
