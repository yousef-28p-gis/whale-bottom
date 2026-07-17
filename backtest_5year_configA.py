#!/usr/bin/env python3 -u
"""Config A: 5-year backtest WITHOUT worst 10 coins"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/cache/5year'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {0,1,3,4,6,12}

# Skip worst 10 coins
SKIP = {'XMR','DASH','BTC','FLOW','AR','COMP','ENJ','1INCH','MATIC','MKR'}

print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN} RSI<25')
print(f'🔍 Config A — skipping worst 10: {",".join(sorted(SKIP))}')
print()

all_trades = []
done = 0

for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith('.json'): continue
    sym = fname.replace('_15m.json','')
    if sym in SKIP:
        print(f'  ⏭️  SKIP {sym}')
        continue
    
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
    if sym_trades > 0 or done % 5 == 0:
        sym_wins = sum(1 for t in all_trades if t['sym']==sym and t['pnl']>0)
        sym_wr = sym_wins/sym_trades*100 if sym_trades>0 else 0
        print(f'  {done:>2}/40 {sym:<8} {sym_trades:>4}ت | WR {sym_wr:.0f}%', flush=True)

if not all_trades:
    print('لا توجد صفقات!')
    exit()

nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
exits=defaultdict(int)
for t in all_trades: exits[t['exit']] += 1

print(f'\n{"="*70}')
print(f'📊 CONFIG A — 40 coins (skip 10 worst) — 5 years — RSI<25')
print(f'{"="*70}')
print(f'📋 Total signals: {len(all_trades)} | 🟢 Wins: {wins} | 🔴 Losses: {len(all_trades)-wins}')
print(f'📈 WR: {wins/len(all_trades)*100:.1f}%')
print(f'🎯 TP={exits.get("TP",0)} 🛑 SL={exits.get("SL",0)} 🐌 TRAIL={exits.get("TRAIL",0)} ⏰ TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print(f'💰 Net PnL Sum: {sum(nets):+.1f}%')
print()

# Portfolio: 2 x 50%
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

print(f'{"="*70}')
print(f'💼 Portfolio — 2 positions × 50% — {years} years')
print(f'{"="*70}')
print(f'✅ Executed trades: {taken} | ⏭️ Skipped: {skipped}')
print(f'📈 Executed WR: {exec_wins/taken*100:.1f}%' if taken>0 else 'N/A')
final_pct = (capital/1000 - 1) * 100
print(f'💼 Final capital: $1000 → ${capital:.0f} ({final_pct:+.1f}%)')
print(f'📉 Max DD: {max_dd:.2f}%')
ann_return = (capital/1000)**(1/years)-1
print(f'📈 Annual return: {ann_return*100:+.1f}%')
print()

# Yearly
print('📅 By year:')
for yr in range(2021, 2027):
    yt = [t for t in exec_trades if t['dt'].year==yr]
    if not yt: continue
    yw = sum(1 for t in yt if t['pnl']>0)
    print(f'  {yr}: {len(yt):>3} trades | WR {yw/len(yt)*100:.0f}%')
