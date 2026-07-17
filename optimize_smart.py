#!/usr/bin/env python3 -u
"""حساب المؤشرات مرة وحدة ثم مقارنة الكونفجات"""
import json, os, sys, numpy as np, pandas as pd
from collections import defaultdict

CACHE_DIR = '/data/trading28/cache/5year'
TRADES_FILE = '/data/trading28/trades_5year.json'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
SKIP_COINS = {'XMR','DASH','BTC','FLOW','AR','COMP','ENJ','1INCH','MATIC','MKR'}

# Check if already computed
if os.path.exists(TRADES_FILE):
    print('⏭️ الملف موجود... بتحميل', flush=True)
    with open(TRADES_FILE) as f:
        all_candidates = json.load(f)
    # Convert string dates back
    for c in all_candidates:
        c['dt'] = pd.Timestamp(c['dt'])
    print(f'✅ {len(all_candidates)} مرشح محمل', flush=True)
else:
    print('🔨 حساب المؤشرات لجميع العملات...', flush=True)
    all_candidates = []
    done = 0
    
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith('.json'): continue
        sym = fname.replace('_15m.json','')
        if sym in SKIP_COINS: continue
        
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
        
        for i in range(50, len(df)-10):
            row = df.iloc[i]
            if not row['entry']: continue
            wv = float(row['whale'])
            if wv < 0.40: continue  # minimum baseline
            if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
            
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
            pump24 = (ep-pb)/pb*100 if pb!=0 else 0
            
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
                'sym': sym, 'dt': str(row['ts']),
                'pnl': pnl, 'exit': exit_,
                'wv': round(wv,4), 'rsi': round(float(row['rsi']),1),
                'p24': round(pump24,2),
                'h': int(row['ts'].hour), 'wd': int(row['ts'].weekday())
            })
        
        done += 1
        if done % 10 == 0:
            print(f'  {done} عملات | {len(all_candidates)} مرشح', flush=True)
    
    # Save
    with open(TRADES_FILE, 'w') as f:
        json.dump(all_candidates, f)
    print(f'✅ تم حفظ {len(all_candidates)} مرشح', flush=True)

# Fix dates
for c in all_candidates:
    c['dt'] = pd.Timestamp(c['dt'])

# Now test configs
print(f'\n{"="*60}')
print(f'📊 مقارنة الكونفجات')
print(f'{"="*60}')
print(f'{"الإعداد":<28} {"صفقات":<8} {"WR":<8} {"محفظة":<10} {"سحب"}')
print('-'*70)

from datetime import timedelta

def test_config(candidates, wm, rs, p24, label):
    filtered = [c for c in candidates if c['wv'] >= wm and c['rsi'] < rs 
                and c['p24'] < p24 and c['wd'] != 3 
                and c['h'] not in BLOCK_HOURS]
    if len(filtered) < 10: return None
    
    trades_sorted = sorted(filtered, key=lambda x: x['dt'])
    capital = 1000.0; peak = 1000.0; max_dd = 0.0
    active = []; taken = 0
    
    for t in trades_sorted:
        dt = t['dt']
        still_active = []
        for exit_dt, cost, pnl_amt in active:
            if dt >= exit_dt:
                capital += cost + pnl_amt
            else:
                still_active.append((exit_dt, cost, pnl_amt))
        active = still_active
        if len(active) >= 2: continue
        pos_size = capital * 0.50
        if capital < pos_size: continue
        pnl_amt = pos_size * t['pnl'] / 100
        capital -= pos_size
        active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
        taken += 1
        
        equity = capital + sum(pc + pd for _, pc, pd in active)
        if equity > peak: peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd: max_dd = dd
    
    for _, cost, pnl_amt in active:
        capital += cost + pnl_amt
    
    wins = sum(1 for t in filtered if t['pnl']>0)
    wr = wins/len(filtered)*100
    
    print(f'{label:<28} {len(filtered):<8} {wr:<8.1f}% ${capital:<9.0f} {round(max_dd,2)}%')
    return capital

configs = [
    (0.50, 25, 0, 'أساسي (بدون 10 أسوأ)'),
    (0.50, 25, 0, 'أساسي'),
    (0.55, 25, 0, '🐋≥0.55'),
    (0.50, 22, 0, '📉 RSI<22'),
    (0.50, 20, 0, '📉 RSI<20'),
    (0.55, 22, 0, '🐋≥0.55 + RSI<22'),
    (0.55, 20, 0, '🐋≥0.55 + RSI<20'),
    (0.50, 25, -1, '📊 pump24<-1%'),
    (0.55, 25, -1, '🐋≥0.55 + pump24<-1%'),
    (0.55, 22, -1, '🔥 الكل: ≥0.55+<22+<-1%'),
]

for wm, rs, p24, label in configs:
    test_config(all_candidates, wm, rs, p24, label)

print(f'\n✅ تم!')
