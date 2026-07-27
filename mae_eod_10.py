#!/usr/bin/env python3
"""تحليل MAE + EOD — 10 عملات سريعة"""
import json, os, sys, numpy as np, pandas as pd

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}

# 10 diverse coins — mix of large cap / mid cap / small cap
COINS = ['ADA', 'ETH', 'SOL', 'DOGE', 'AVAX', 'LINK', 'DOT', 'ATOM', 'GRT', 'SAND']
files = [f'{c}_15m.json' for c in COINS if os.path.exists(f'{CACHE_DIR}/{c}_15m.json')]

mae_list = []
eod_list = []
eod_win = []
eod_loss = []
exit_counts = {}

for idx, fname in enumerate(files):
    sym = fname.replace('_15m.json','')
    if sym in BLACKLIST: continue
    
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
        
        exit_counts[exit_] = exit_counts.get(exit_, 0) + 1
        sym_trades += 1
    
    if sym_trades > 0:
        sym_tp = exit_counts.get('TP', 0)
        print(f'  {sym:<8} {sym_trades:>3}ت | TP={sym_tp}', flush=True)

print(f'\n{"="*60}')
print(f'📊 تحليل MAE + EOD — {len(files)} عملات')
print(f'{"="*60}')
print(f'📋 إجمالي: {len(eod_list)} صفقة')
print(f'📊 مخارج: {exit_counts}')

print(f'\n🐋 السؤال 2: كم تنزل العملة قبل الهدف؟')
print(f'   صفقات TP: {len(mae_list)}')
if mae_list:
    arr = np.array(mae_list)
    print(f'   متوسط النزول: {arr.mean():.3f}%')
    print(f'   وسيط: {np.median(arr):.3f}%')
    print(f'   أقصى نزول: {arr.min():.3f}%  |  أقل نزول: {arr.max():.3f}%')
    
    bins = [(-0.25, 0), (-0.5, -0.25), (-1.0, -0.5), (-1.5, -1.0), (-2.0, -1.5), (-5, -2.0)]
    print(f'   توزيع:')
    for lo, hi in bins:
        cnt = ((arr > lo) & (arr <= hi)).sum()
        pct = cnt / len(arr) * 100
        bar = '█' * max(1, int(pct))
        print(f'   {lo:+.2f}% → {hi:+.2f}%: {cnt:>3} ({pct:>5.1f}%) {bar}')

print(f'\n📅 السؤال 3: ارتفاع نهاية اليوم (EOD ~24h)')
if eod_list:
    arr = np.array(eod_list)
    print(f'   كل الصفقات ({len(arr)}):')
    print(f'     متوسط: {arr.mean():.2f}%  |  وسيط: {np.median(arr):.2f}%')
    pos = (arr > 0).sum()
    print(f'     موجبة EOD: {pos}/{len(arr)} ({pos/len(arr)*100:.1f}%)')

if eod_win:
    arr = np.array(eod_win)
    print(f'\n   📈 لصفقات الرابحة ({len(arr)}): متوسط EOD {arr.mean():.2f}%  |  وسيط {np.median(arr):.2f}%')

if eod_loss:
    arr = np.array(eod_loss)
    print(f'\n   📉 لصفقات الخاسرة ({len(arr)}): متوسط EOD {arr.mean():.2f}%  |  وسيط {np.median(arr):.2f}%')
    pos = (arr > 0).sum()
    print(f'   منها ارتفعت EOD: {pos}/{len(arr)} ({pos/len(arr)*100:.1f}%)')

print('\n✅ تم')
