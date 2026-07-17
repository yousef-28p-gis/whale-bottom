#!/usr/bin/env python3 -u
"""باك تيست مع جميع الفلاتر — صفقتين × 50%"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/cache/ohlcv'
MONTHS = ['2026-04', '2026-05', '2026-06']
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20

BLOCK_COINS = {'ENA','ZEC','ALLO','ZRO','CHIP','EDEN'}

all_trades = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith('.json'): continue
    parts = fname.replace('.json','').rsplit('_',1)
    if len(parts) != 2: continue
    sym, mon = parts
    if mon not in MONTHS: continue
    if sym in BLOCK_COINS: continue
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        try: data = json.load(f)
        except: continue
    
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
    
    for i in range(50, len(df)-10):
        row = df.iloc[i]
        if row['ts'].month < 4: continue
        if not (row['entry'] and float(row['whale']) >= WHALE_MIN): continue
        if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
        
        # ⛔ حظر الخميس
        if row['ts'].weekday() == 3: continue  # Monday=0, Thursday=3
        # ⛔ حظر ساعات 01, 03, 06, 12
        if row['ts'].hour in (1, 3, 6, 12): continue
        
        ps = max(0, i-96); pb = float(df.iloc[ps]['close'])
        ep = float(row['close'])
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
        all_trades.append({'dt': row['ts'], 'pnl': pnl, 'exit': exit_, 'sym': sym})

# ── Results ──
nets = [t['pnl'] for t in all_trades]
wins = sum(1 for n in nets if n > 0)
exits = defaultdict(int)
for t in all_trades: exits[t['exit']] += 1

print(f'📊 إجمالي الإشارات المؤهلة: {len(all_trades)}')
print(f'🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades)-wins}')
print(f'📈 WR: {wins/len(all_trades)*100:.1f}%' if all_trades else 'N/A')
print(f'🎯 TP={exits.get("TP",0)} 🛑 SL={exits.get("SL",0)} 🐌 TRAIL={exits.get("TRAIL",0)} ⏰ TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print()

# ── Portfolio: 2 × 50% ──
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
monthly_return = (capital / 1000) ** (1 / 3) - 1

print(f'{"="*60}')
print(f'🐋 صفقتين × 50% مع جميع الفلاتر')
print(f'{"="*60}')
print(f'⚙️ WHALE≥0.50 | ⛔ خميس + ساعات 1,3,6,12 | ⛔ 6 عملات محظورة')
print(f'📋 إشارات مؤهلة: {len(all_trades)}')
print(f'✅ منفذة: {taken} | ⏭️ متخطية: {skipped}')
print(f'🟢 رابحة: {exec_wins} | 🔴 خاسرة: {taken-exec_wins}')
print(f'📈 WR: {exec_wins/taken*100:.1f}%' if taken > 0 else 'N/A')
print(f'💼 محفظة: $1000 → ${capital:.0f} ({capital/10-100:+.1f}%)')
print(f'📈 عائد شهري: {monthly_return*100:+.1f}%')
print(f'📉 أقصى سحب: {max_dd:.2f}%')

months_ar = {4:'أبريل',5:'مايو',6:'يونيو'}
for m in [4,5,6]:
    mt = [t for t in exec_trades if t['dt'].month==m]
    if not mt: continue
    mw = sum(1 for t in mt if t['pnl']>0)
    print(f'  {months_ar[m]}: {len(mt)} صفقة | WR {mw/len(mt)*100:.0f}%')
