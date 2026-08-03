#!/usr/bin/env python3
"""Per-coin stats from EXECUTED trades only"""
import json, numpy as np, os, time, gc, pandas as pd

COMM = 0.20; TF_MIN = 3; MAX_POS = 2
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

with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
EXCLUDE = {'ETH','BTC','TRX','XRP','QI','LSK','GLMR','XTZ','YFI',
           'TLM','0G','LA','DYM','VANRY','SENT','VET','COOKIE','HEI','ACT','CKB','AR','RSR','AXS','XEC'}
COINS = [c for c in COINS if c not in EXCLUDE]

all_trades = []
processed = 0; t0 = time.time()

for coin in COINS:
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw = json.load(f)
    if len(raw) < 200: del raw; continue
    df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw; df = compute_indicators(df)
    close_arr = df['close'].values
    ts_arr = df['ts'].values.astype('datetime64[ns]').astype('int64')
    idxs = find_signals(df)
    if len(idxs) == 0: del df; continue
    max_bars = int(MAX_H*60/TF_MIN)
    tp_r=1+TP/100; sl_r=1-SL/100; tr_r=1-TRAIL/100
    active=[]; sig_map=dict(zip(idxs, close_arr[idxs]))
    for i in range(len(df)):
        cur=close_arr[i]
        if i in sig_map:
            active.append({'symbol':coin,'entry':sig_map[i],'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                'pl_ok':False,'peak':sig_map[i],'trail':sig_map[i],'entry_i':i,'entry_ns':int(ts_arr[i])})
        for j in range(len(active)-1,-1,-1):
            p=active[j]; e=p['entry']; bh=i-p['entry_i']
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
                pl_p=e+(p['tp']-e)*(PL/100)
                if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*tr_r
    del df; gc.collect(); processed+=1

all_trades.sort(key=lambda t: t['entry_ns'])

# Global MAX_POS=2
equity=1000; active_slots=[None,None]; executed_trades=[]
for t in all_trades:
    en=t['entry_ns']; ex=t['exit_ns']; pp=t['pnl']
    for s in range(2):
        if active_slots[s] and active_slots[s][0]<=en:
            active_slots[s]=None
    free=-1
    for s in range(2):
        if not active_slots[s]: free=s; break
    if free==-1: continue
    active_slots[free]=(ex,pp); executed_trades.append(t)

# Per-coin stats
from collections import defaultdict
coin_stats = defaultdict(lambda: {'trades':0,'pnl_sum':0.0,'wins':0,'losses':0})
for t in executed_trades:
    c = t['symbol']
    coin_stats[c]['trades'] += 1
    coin_stats[c]['pnl_sum'] += t['pnl']
    if t['pnl'] > 0: coin_stats[c]['wins'] += 1
    else: coin_stats[c]['losses'] += 1

per_coin = []
for c, s in coin_stats.items():
    wr = s['wins']/s['trades']*100 if s['trades'] else 0
    per_coin.append((c, s['trades'], s['wins'], s['losses'], wr, round(s['pnl_sum'],2)))
per_coin.sort(key=lambda x: x[5])

print(f"\n{'='*70}")
print(f"عدد الصفقات المنفذة: {len(executed_trades):,} عبر {len(per_coin)} عملة")
print(f"{'='*70}")

lt10 = [p for p in per_coin if p[5] < 10]
lt5 = [p for p in per_coin if p[5] < 5]
neg = [p for p in per_coin if p[5] < 0]

print(f"\nعملات صافي ربحها أقل من 10%: {len(lt10)} عملة")
for p in lt10[:30]:
    print(f"  {p[0]:12s} | {p[1]:3d} صفقة | WR {p[4]:5.1f}% | صافي {p[5]:+.2f}%")

print(f"\nعملات صافي ربحها أقل من 5%: {len(lt5)} عملة")
for p in lt5[:30]:
    print(f"  {p[0]:12s} | {p[1]:3d} صفقة | WR {p[4]:5.1f}% | صافي {p[5]:+.2f}%")

print(f"\nعملات خاسرة: {len(neg)} عملة")
for p in neg:
    print(f"  {p[0]:12s} | {p[1]:3d} صفقة | WR {p[4]:5.1f}% | صافي {p[5]:+.2f}%")
