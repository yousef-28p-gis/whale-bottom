#!/usr/bin/env python3 -u
"""باك تيست 5 سنوات — عملات حلال فقط — حوت القاع"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/cache/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}

print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN} RSI<25')
print(f'🔍 عملات حلال | 5 سنوات | صفقتين×50%')
print()

all_trades = []
done = 0
coin_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

for fname in coin_files:
    sym = fname.replace('_15m.json','')
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
    # Whale
    LB=30
    df['lo']=df['low'].rolling(LB).min()
    df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
    df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
    df['hi']=df['sm'].rolling(LB).max()
    df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
    df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
    df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
    df['wf']=df['whale'].rolling(2).mean(); df['ws']=df['whale'].rolling(5).mean()
    df['wp']=df['whale'].rolling(50).max()
    df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
    df['vma']=df['volume'].rolling(20).mean()
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100/(1+rs))
    
    sym_trades = 0
    for i in range(50, len(df)-10):
        row = df.iloc[i]
        if not row['entry']: continue
        if float(row['whale']) < WHALE_MIN: continue
        if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25: continue
        if row['ts'].weekday() == 3: continue
        if row['ts'].hour in BLOCK_HOURS: continue
        ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
        if (ep-pb)/pb*100 >= 0: continue
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
            if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
            if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
        else: pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
        all_trades.append({'sym':sym, 'dt':row['ts'], 'pnl':pnl, 'exit':exit_})
        sym_trades += 1
    
    done += 1
    if sym_trades > 0 or done % 10 == 0:
        sym_wins = sum(1 for t in all_trades if t['sym']==sym and t['pnl']>0)
        sym_wr = sym_wins/sym_trades*100 if sym_trades>0 else 0
        print(f'  {done:>3}/{len(coin_files)} {sym:<12} {sym_trades:>4}ت | WR {sym_wr:.0f}%', flush=True)

if not all_trades:
    print('لا توجد صفقات!')
    exit()

# Stats
nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
exits=defaultdict(int)
for t in all_trades: exits[t['exit']] += 1

print(f'\n{"="*60}')
print(f'📊 عملات حلال — 5 سنوات — RSI<25')
print(f'{"="*60}')
print(f'📋 إشارات: {len(all_trades)} | 🟢 {wins} | 🔴 {len(all_trades)-wins}')
print(f'📈 WR: {wins/len(all_trades)*100:.1f}%')
print(f'🎯 TP={exits.get("TP",0)} 🛑 SL={exits.get("SL",0)} 🐌 TRAIL={exits.get("TRAIL",0)} ⏰ TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print(f'💰 صافي: {sum(nets):+.1f}%')
print()

# Per coin stats
coin_stats = defaultdict(lambda: {'t':0,'w':0,'pnl':0.0})
for t in all_trades:
    c = coin_stats[t['sym']]
    c['t'] += 1; c['pnl'] += t['pnl']
    if t['pnl']>0: c['w']+=1

print('🏆 أفضل 15 عملة:')
for sym, c in sorted(coin_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:15]:
    wr = c['w']/c['t']*100
    print(f'  {sym:<12} {c["t"]:>3}ت | WR {wr:.0f}% | صافي {c["pnl"]:+.1f}%')

print('\n💀 أسوأ 15 عملة:')
for sym, c in sorted(coin_stats.items(), key=lambda x: x[1]['pnl'])[:15]:
    wr = c['w']/c['t']*100
    print(f'  {sym:<12} {c["t"]:>3}ت | WR {wr:.0f}% | صافي {c["pnl"]:+.1f}%')

# Portfolio: 2 × 50%
trades_sorted = sorted(all_trades, key=lambda x: x['dt'])
capital = 1000.0; peak = 1000.0; max_dd = 0.0
active = []; skipped = 0; taken = 0; exec_trades = []

for t in trades_sorted:
    dt = t['dt']
    still_active = []
    for exit_dt, cost, pnl_amt in active:
        if dt >= exit_dt:
            capital += cost + pnl_amt
        else:
            still_active.append((exit_dt, cost, pnl_amt))
    active = still_active
    
    if len(active) >= 2:
        skipped += 1; continue
    pos_size = capital * 0.50
    if capital < pos_size:
        skipped += 1; continue
    pnl_amt = pos_size * t['pnl'] / 100
    capital -= pos_size
    active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
    taken += 1
    exec_trades.append(t)
    
    equity = capital + sum(pc + pd for _, pc, pd in active)
    if equity > peak: peak = equity
    dd = (equity - peak) / peak * 100
    if dd < max_dd: max_dd = dd

for _, cost, pnl_amt in active:
    capital += cost + pnl_amt

exec_nets = [t['pnl'] for t in exec_trades]
exec_wins = sum(1 for n in exec_nets if n > 0)
years = 5

print(f'\n{"="*60}')
print(f'💼 محفظة — صفقتين × 50% — {years} سنوات')
print(f'{"="*60}')
print(f'✅ منفذة: {taken} | ⏭️ متخطية: {skipped}')
print(f'📈 WR منفذة: {exec_wins/taken*100:.1f}%' if taken>0 else 'N/A')
print(f'💼 محفظة: $1000 → ${capital:.0f} ({capital/10-100:+.1f}%)')
print(f'📉 سحب: {max_dd:.2f}%')
ann_return = (capital/1000)**(1/years)-1 if capital > 0 else -1
print(f'📈 عائد سنوي: {ann_return*100:+.1f}%')

# Yearly
print('\nحسب السنة:')
for yr in [2021,2022,2023,2024,2025,2026]:
    yt = [t for t in exec_trades if t['dt'].year==yr]
    if not yt: continue
    yw = sum(1 for t in yt if t['pnl']>0)
    yp = sum(t['pnl'] for t in yt)
    print(f'  {yr}: {len(yt):>3} صفقة | WR {yw/len(yt)*100:.0f}% | صافي {yp:+.1f}%')
