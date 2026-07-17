#!/usr/bin/env python3 -u
"""Grid search لرفع WR فوق 80%"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict

CACHE_DIR = '/data/trading28/cache/ohlcv'
MONTHS = ['2026-04', '2026-05', '2026-06']
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; COMM=0.20

# Build all trade candidates once
print('⏳ تحميل البيانات...', flush=True)

all_candidates = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith('.json'): continue
    parts = fname.replace('.json','').rsplit('_',1)
    if len(parts) != 2: continue
    sym, mon = parts
    if mon not in MONTHS: continue
    
    with open(f'{CACHE_DIR}/{fname}') as f:
        try: data = json.load(f)
        except: continue
    
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 500: continue
    
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
        if not row['entry']: continue
        
        wv = float(row['whale'])
        if wv < 0.40: continue  # base minimum
        
        wn = float(df.iloc[i+1]['whale']) if i+1 < len(df) else 0
        ps = max(0, i-96); pb = float(df.iloc[ps]['close'])
        ep = float(row['close'])
        p24 = (ep-pb)/pb*100 if pb != 0 else 0
        if p24 >= 0: continue  # base pump24 filter
        
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
        
        all_candidates.append({
            'sym': sym, 'dt': row['ts'], 'pnl': pnl,
            'whale_val': wv, 'whale_next': wn, 'pump24': p24,
            'hour': row['ts'].hour, 'weekday': row['ts'].weekday()
        })

print(f'✅ {len(all_candidates)} مرشح (whale≥0.40 + pump24<0)', flush=True)
print()

# ── Grid search ──
configs = []
for wm in [0.40, 0.45, 0.50, 0.55]:
    for wn_max in [0.35, 0.25, 0.20, 0.15]:
        for p24_max in [-0.5, -1.0, -2.0, -3.0]:
            for block_bad in [False, True]:
                filtered = []
                for c in all_candidates:
                    if c['whale_val'] < wm: continue
                    if c['whale_next'] > wn_max: continue
                    if c['pump24'] > p24_max: continue
                    if block_bad:
                        if c['weekday'] == 3: continue  # Thu
                        if c['hour'] in (1,3,6,12,0,4): continue
                    filtered.append(c)
                
                if len(filtered) < 100: continue
                
                wins = sum(1 for c in filtered if c['pnl'] > 0)
                wr = wins / len(filtered) * 100
                if wr >= 78:
                    configs.append({
                        'wm': wm, 'wn': wn_max, 'p24': p24_max,
                        'block': block_bad,
                        'n': len(filtered), 'wr': wr,
                        'wins': wins
                    })

configs.sort(key=lambda x: x['wr'], reverse=True)

print(f'🔍 {len(configs)} كونفج WR ≥ 78%')
print()
print(f'{"حوت":<6} {"شمعة":<7} {"pump24":<8} {"حظر":<5} {"صفقات":<7} {"WR":<7}')
print('-'*50)
for c in configs[:25]:
    block = '✓' if c['block'] else '-'
    print(f'{c["wm"]:.2f}   <{c["wn"]:.2f}    <{c["p24"]:.1f}%  {block:<5} {c["n"]:<7} {c["wr"]:.1f}%')

# Best by trade count within WR ranges
print()
print('='*60)
print('🏆 أفضل كونفج لكل مستوى WR:')
for target_wr in [80, 82, 84]:
    best = None
    for c in configs:
        if c['wr'] >= target_wr:
            if best is None or c['n'] > best['n']:
                best = c
    if best:
        block = 'نعم' if best['block'] else 'لا'
        print(f'  🎯 WR≥{target_wr}%: حوت≥{best["wm"]} | شمعة<{best["wn"]} | pump24<{best["p24"]}% | حظر={block}')
        print(f'     {best["n"]} صفقة | WR {best["wr"]:.1f}% | {best["wins"]} رابحة')
