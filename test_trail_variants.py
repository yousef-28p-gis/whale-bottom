#!/usr/bin/env python3
"""مقارنة 6 حلول للتريل — 15 عملة"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}
COINS = ['ADA','ETH','SOL','DOGE','AVAX','LINK','DOT','ATOM','GRT','SAND','MATIC','NEAR','FIL','AR','FET']

# المتغيرات اللي بدنا نجربها
variants = [
    {'name': '0. حالي (15m تريل 0.1%)', 'tp': TP, 'sl': SL, 'pl': PL, 'trail': TRAIL, 'partial': False},
    {'name': '1. بدون تريل',               'tp': TP, 'sl': SL, 'pl': 100, 'trail': 0, 'partial': False},
    {'name': '2. تريل 0.3%',               'tp': TP, 'sl': SL, 'pl': PL, 'trail': 0.30, 'partial': False},
    {'name': '2. تريل 0.5%',               'tp': TP, 'sl': SL, 'pl': PL, 'trail': 0.50, 'partial': False},
    {'name': '3. PL=50% (تأخير التريل)',   'tp': TP, 'sl': SL, 'pl': 50, 'trail': TRAIL, 'partial': False},
    {'name': '3. PL=70% (تأخير التريل)',   'tp': TP, 'sl': SL, 'pl': 70, 'trail': TRAIL, 'partial': False},
    {'name': '4. نصف جزئي (50% TP)',        'tp': TP, 'sl': SL, 'pl': PL, 'trail': TRAIL, 'partial': True},
    {'name': '5. بدون تريل + TP=5%',        'tp': 5.0, 'sl': SL, 'pl': 100, 'trail': 0, 'partial': False},
    {'name': '5. بدون تريل + TP=4%',        'tp': 4.0, 'sl': SL, 'pl': 100, 'trail': 0, 'partial': False},
]

results = {}

for var in variants:
    vname = var['name']
    print(f'\n⏳ {vname}...', flush=True)
    
    v_tp = var['tp']
    v_sl = var['sl']
    v_pl = var['pl']
    v_trail = var['trail']
    v_partial = var['partial']
    
    # PL=100 means no trail (never triggers)
    if v_pl >= 100:
        use_trail = False
        v_pl_val = 100
    else:
        use_trail = v_trail > 0
        v_pl_val = v_pl
    
    all_pnl = []
    exit_counts = defaultdict(int)
    
    for sym in COINS:
        fp = f'{CACHE_DIR}/{sym}_15m.json'
        if not os.path.exists(fp): continue
        if sym in BLACKLIST: continue
        
        with open(fp) as f:
            data = json.load(f)
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
        delta = df['close'].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100/(1+rs))
        
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
            if i+1 >= len(df): continue
            if float(df.iloc[i+1]['close']) <= float(df.iloc[i+1]['open']): continue
            
            tp_p=ep*(1+v_tp/100); sl_p=ep*(1-v_sl/100)
            pl_p=ep+(tp_p-ep)*(v_pl_val/100)
            pl_trig=False; peak=ep; trail_p=0
            
            if v_partial:
                # Half at TP, half runs without trail
                for k in range(i+1, len(df)):
                    cur=float(df.iloc[k]['close']); h=(k-i)*0.25
                    if h>MH:
                        half1 = 0
                        for k2 in range(i+1, k):
                            if float(df.iloc[k2]['close']) >= tp_p:
                                half1 = round(TP-COMM, 4)
                                break
                        half2 = round((cur-ep)/ep*100-COMM, 4)
                        pnl = round((half1 + half2) / 2, 4)
                        exit_='PARTIAL_TIME'; break
                    if cur>=tp_p:
                        # Half exits at TP, rest continues
                        half1_pnl = round(TP-COMM, 4)
                        # Second half: no trail, TP only
                        p2 = None
                        for k2 in range(k+1, len(df)):
                            c2=float(df.iloc[k2]['close']); h2=(k2-i)*0.25
                            if h2>MH: p2=round((c2-ep)/ep*100-COMM,4); break
                            if c2>=tp_p: p2=round(TP-COMM,4); break
                            if c2<=sl_p: p2=round(-SL-COMM,4); break
                        if p2 is None: p2=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
                        pnl = round((half1_pnl + p2) / 2, 4)
                        exit_='PARTIAL_TP'; break
                    if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
                else:
                    pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
            else:
                # Normal / no-trail
                for k in range(i+1, len(df)):
                    cur=float(df.iloc[k]['close']); h=(k-i)*0.25
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(v_tp-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-v_sl-COMM,4); exit_='SL'; break
                    if use_trail:
                        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-v_trail/100)
                        if pl_trig:
                            if cur>peak: peak=cur; trail_p=cur*(1-v_trail/100)
                            if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
                else:
                    pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
            
            all_pnl.append(pnl)
            exit_counts[exit_] += 1
    
    arr = np.array(all_pnl)
    wins = (arr > 0).sum()
    total = len(arr)
    results[vname] = {
        'trades': total, 'wins': int(wins), 'sum': float(arr.sum()),
        'mean': float(arr.mean()), 'exits': dict(exit_counts),
        'std': float(arr.std())
    }
    print(f'   ✅ {total}ت | WR {wins/total*100:.1f}% | مجموع {arr.sum():+.1f}% | متوسط {arr.mean():+.3f}%', flush=True)

print(f'\n{"="*75}')
print(f'📊 مقارنة 6 حلول للتريل — 15 عملة')
print(f'{"="*75}')
print(f'{"الحل":<35} {"صفقات":>5} {"WR":>6} {"مجموع":>8} {"متوسط":>7} {"TP":>4} {"SL":>4} {"TRAIL":>5} {"TIME":>5}')
print(f'{"-"*75}')

for vname, r in results.items():
    e = r['exits']
    tp_c = e.get('TP', 0) + e.get('PARTIAL_TP', 0)
    sl_c = e.get('SL', 0)
    tr_c = e.get('TRAIL', 0)
    tm_c = e.get('TIME', 0) + e.get('EOD', 0) + e.get('PARTIAL_TIME', 0)
    wr = r['wins']/r['trades']*100
    print(f'{vname:<35} {r["trades"]:>5} {wr:>5.1f}% {r["sum"]:>+7.1f}% {r["mean"]:>+6.3f}% {tp_c:>4} {sl_c:>4} {tr_c:>5} {tm_c:>5}')

print(f'\n✅ تم')
