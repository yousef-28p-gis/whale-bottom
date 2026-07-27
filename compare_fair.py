#!/usr/bin/env python3
"""مقارنة عادلة: 15m vs 1h × مع/بدون فلتر 3"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}

BLACKLIST = {
    'QTUM', 'ZRO', 'IOTX', 'DYM', 'DGB', 'SAPIEN', 'XLM',
    'EDU', 'BTC', 'INIT', 'PARTI', '0G', 'ROBO', 'PYTH', 'ANKR'
}

configs = [
    ('15m بدون فلتر3',   1, False),
    ('15m مع فلتر3',     1, True),
    ('1h بدون فلتر3',    4, False),
    ('1h مع فلتر3',      4, True),
]

results = {}
all_trades_by_cfg = {c[0]: [] for c in configs}

coin_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN} RSI<25')
print(f'🚫 مستبعدة: {", ".join(sorted(BLACKLIST))}')
print(f'🔄 4 باك تيستات متزامنة...')
print()

for fname in coin_files:
    sym = fname.replace('_15m.json','')
    if sym in BLACKLIST:
        continue
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
    # Indicators (shared)
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
    
    for idx in range(50, len(df)-10):
        row = df.iloc[idx]
        if not row['entry']: continue
        if float(row['whale']) < WHALE_MIN: continue
        if idx+1 < len(df) and float(df.iloc[idx+1]['whale']) >= 0.35: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25: continue
        if row['ts'].weekday() == 3: continue
        if row['ts'].hour in BLOCK_HOURS: continue
        ps=max(0,idx-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
        if (ep-pb)/pb*100 >= 0: continue
        
        has_confirm = (idx+1 < len(df) and float(df.iloc[idx+1]['close']) > float(df.iloc[idx+1]['open']))
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        
        for cfg_name, step, use_filter3 in configs:
            if use_filter3 and not has_confirm:
                continue
            
            pl_trig=False; peak=ep; trail_p=0
            pnl = None; exit_ = 'EOD'
            
            if step == 1:
                # 15m — check every candle
                for k in range(idx+1, len(df)):
                    cur=float(df.iloc[k]['close']); h=(k-idx)*0.25
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
                    if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                    if pl_trig:
                        if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                        if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
            else:
                # 1h — check every 4th candle
                k = idx + step
                while k < len(df):
                    cur=float(df.iloc[k]['close']); h=(k-idx)*0.25
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
                    if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                    if pl_trig:
                        if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                        if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
                    k += step
            
            if pnl is None:
                pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
            
            all_trades_by_cfg[cfg_name].append({'sym':sym, 'dt':row['ts'], 'pnl':pnl, 'exit':exit_})
    
    # Print progress
    sym_trades = sum(1 for t in all_trades_by_cfg['15m بدون فلتر3'] if t['sym']==sym)
    if sym_trades > 0 or len([f for f in coin_files if f > fname]) % 10 == 0:
        done = sum(1 for f in coin_files if f <= fname and f.replace('_15m.json','') not in BLACKLIST)
        total = len(coin_files) - len([f for f in coin_files if f.replace('_15m.json','') in BLACKLIST])
        print(f'  {done:>3}/{total} {sym:<12} {sym_trades:>4}ت', flush=True)

print(f'\n{"="*75}')
print(f'📊 مقارنة عادلة — نفس الإشارات، نفس كل شي')
print(f'{"="*75}')
print(f'{"الباك تيست":<25} {"صفقات":>6} {"رابحة":>6} {"WR":>7} {"صافي":>9} {"محفظة":>10} {"سحب":>7} {"سنوي":>7}')
print(f'{"-"*75}')

years = 5

for cfg_name in ['15m بدون فلتر3', '15m مع فلتر3', '1h بدون فلتر3', '1h مع فلتر3']:
    trades = all_trades_by_cfg[cfg_name]
    nets = [t['pnl'] for t in trades]
    wins = sum(1 for n in nets if n > 0)
    wr = wins/len(trades)*100 if trades else 0
    total_pnl = sum(nets)
    
    # Portfolio
    trades_sorted = sorted(trades, key=lambda x: x['dt'])
    capital = 1000.0; peak_cap = 1000.0; max_dd = 0.0
    active = []; skipped = 0; taken = 0
    
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
        
        equity = capital + sum(pc + pd for _, pc, pd in active)
        if equity > peak_cap: peak_cap = equity
        dd = (equity - peak_cap) / peak_cap * 100
        if dd < max_dd: max_dd = dd
    
    for _, cost, pnl_amt in active:
        capital += cost + pnl_amt
    
    ann_return = (capital/1000)**(1/years)-1 if capital > 0 else -1
    
    print(f'{cfg_name:<25} {len(trades):>6} {wins:>6} {wr:>6.1f}% {total_pnl:>+8.1f}% ${capital:>9,.0f} {max_dd:>6.2f}% {ann_return*100:>+6.1f}%')

print(f'\n✅ تم')
