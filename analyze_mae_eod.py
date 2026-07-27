#!/usr/bin/env python3
"""تحليل MAE + EOD من الباك تيست"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}

mae_list = []
eod_list = []
eod_win = []
eod_loss = []

done = 0
coin_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

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
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        exit_ = 'EOD'
        lowest_close = ep
        
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if cur < lowest_close:
                lowest_close = cur
            if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
            if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
            if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
        
        if exit_ == 'TP':
            mae = round((lowest_close - ep) / ep * 100, 4)
            mae_list.append(mae)
        
        eod_idx = min(i + 96, len(df) - 1)
        eod_close = float(df.iloc[eod_idx]['close'])
        eod_return = round((eod_close - ep) / ep * 100, 4)
        eod_list.append(eod_return)
        if pnl > 0:
            eod_win.append(eod_return)
        else:
            eod_loss.append(eod_return)
    
    done += 1
    if done % 20 == 0:
        print(f'⏳ {done}/{len(coin_files)} coins... TP={len(mae_list)} trades', flush=True)

print(f'\n{"="*60}')
print(f'📊 تحليل MAE + EOD')
print(f'{"="*60}')

print(f'\n🐋 MAE للصفقات اللي ضربت هدف (TP):')
print(f'   عدد: {len(mae_list)}')
if mae_list:
    arr = np.array(mae_list)
    print(f'   متوسط النزول: {arr.mean():.2f}%')
    print(f'   وسيط: {np.median(arr):.2f}%')
    print(f'   أقل نزول: {arr.max():.2f}%')
    print(f'   أقصى نزول: {arr.min():.2f}%')
    print(f'   انحراف: {arr.std():.2f}%')
    for p in [10, 25, 50, 75, 90]:
        print(f'   P{p}: {np.percentile(arr, p):.2f}%')
    
    print(f'\n   توزيع النزول:')
    bins_ranges = [(-0.5, 0), (-1.0, -0.5), (-2.0, -1.0), (-3.0, -2.0), (-10, -3.0)]
    for lo, hi in bins_ranges:
        cnt = ((arr > lo) & (arr <= hi)).sum()
        pct = cnt / len(arr) * 100
        print(f'   {lo:+.1f}% to {hi:+.1f}%: {cnt:>4} ({pct:>5.1f}%)')

print(f'\n📅 أداء EOD (~24h من الدخول):')
if eod_list:
    arr = np.array(eod_list)
    print(f'   كل الصفقات ({len(arr)}): متوسط {arr.mean():.2f}% | وسيط {np.median(arr):.2f}%')
    pos = (arr > 0).sum()
    print(f'   موجبة EOD: {pos}/{len(arr)} ({pos/len(arr)*100:.1f}%)')

if eod_win:
    arr = np.array(eod_win)
    print(f'   📈 صفقات رابحة ({len(arr)}): متوسط EOD {arr.mean():.2f}% | وسيط {np.median(arr):.2f}%')

if eod_loss:
    arr = np.array(eod_loss)
    print(f'   📉 صفقات خاسرة ({len(arr)}): متوسط EOD {arr.mean():.2f}% | وسيط {np.median(arr):.2f}%')

print('\n✅ تم')
