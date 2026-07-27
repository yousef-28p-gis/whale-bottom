#!/usr/bin/env python3
"""تحليل: أمر حد -0.2% من سعر الإشارة"""
import json, os, numpy as np, pandas as pd

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}
LIMIT_OFFSET = -0.2  # حد أقل 0.2%

# 15 عملة متنوعة
COINS = ['ADA','ETH','SOL','DOGE','AVAX','LINK','DOT','ATOM','GRT','SAND','MATIC','NEAR','FIL','AR','FET']

results = {'entered': 0, 'missed': 0, 'entered_win': 0, 'entered_loss': 0,
           'missed_win': 0, 'missed_loss': 0}
pnl_normal = []
pnl_limit = []

for sym in COINS:
    fname = f'{sym}_15m.json'
    if not os.path.exists(f'{CACHE_DIR}/{fname}'): continue
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
        
        limit_price = ep * (1 + LIMIT_OFFSET / 100)
        
        # Simulate normal entry (market at close)
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0
        
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if h>MH: normal_pnl=round((cur-ep)/ep*100-COMM,4); break
            if cur>=tp_p: normal_pnl=round(TP-COMM,4); break
            if cur<=sl_p: normal_pnl=round(-SL-COMM,4); break
            if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p: normal_pnl=round((trail_p-ep)/ep*100-COMM,4); break
        
        # Check if limit order fills
        # Does price dip to limit_price before exit?
        limit_filled = False
        limit_entry_idx = None
        
        for k in range(i+1, len(df)):
            cur_low = float(df.iloc[k]['low'])
            cur_close = float(df.iloc[k]['close'])
            h=(k-i)*0.25
            if h>MH: break
            if cur_close >= tp_p: break  # hit TP before limit fill
            if cur_close <= sl_p: break  # hit SL before limit fill
            
            if cur_low <= limit_price:
                limit_filled = True
                limit_entry_idx = k
                break
        
        if not limit_filled:
            # Missed the trade
            results['missed'] += 1
            if normal_pnl > 0:
                results['missed_win'] += 1
            else:
                results['missed_loss'] += 1
            pnl_normal.append(normal_pnl)
            pnl_limit.append(0)  # no trade
            continue
        
        # Limit filled — simulate from limit entry
        limit_ep = limit_price
        results['entered'] += 1
        
        lt_tp = limit_ep*(1+TP/100); lt_sl = limit_ep*(1-SL/100)
        lt_pl = limit_ep+(lt_tp-limit_ep)*(PL/100)
        l_pl_trig=False; l_peak=limit_ep; l_trail=0
        
        for k in range(limit_entry_idx+1, len(df)):
            cur=float(df.iloc[k]['close']); h=(k-i)*0.25
            if h>MH: limit_pnl=round((cur-limit_ep)/limit_ep*100-COMM,4); break
            if cur>=lt_tp: limit_pnl=round(TP-COMM,4); break
            if cur<=lt_sl: limit_pnl=round(-SL-COMM,4); break
            if not l_pl_trig and cur>=lt_pl: l_pl_trig=True; l_peak=cur; l_trail=cur*(1-TRAIL/100)
            if l_pl_trig:
                if cur>l_peak: l_peak=cur; l_trail=cur*(1-TRAIL/100)
                if cur<=l_trail: limit_pnl=round((l_trail-limit_ep)/limit_ep*100-COMM,4); break
        
        pnl_normal.append(normal_pnl)
        pnl_limit.append(limit_pnl)
        
        if limit_pnl > 0:
            results['entered_win'] += 1
        else:
            results['entered_loss'] += 1

print(f'{"="*60}')
print(f'📊 أمر حد -0.2% — تحليل {len(COINS)} عملة')
print(f'{"="*60}')
print()

total = results['entered'] + results['missed']
print(f'📋 إجمالي الإشارات: {total}')
print()
print(f'🟢 دخلنا الصفقة: {results["entered"]} ({results["entered"]/total*100:.1f}%)')
print(f'   منها رابحة: {results["entered_win"]} | خاسرة: {results["entered_loss"]}')
if results['entered'] > 0:
    print(f'   WR: {results["entered_win"]/results["entered"]*100:.1f}%')
print()
print(f'🔴 فاتتنا الصفقة: {results["missed"]} ({results["missed"]/total*100:.1f}%)')
print(f'   كانت راح تربح: {results["missed_win"]} | كانت راح تخسر: {results["missed_loss"]}')

if pnl_normal:
    arr_n = np.array(pnl_normal)
    arr_l = np.array(pnl_limit)
    print(f'\n💰 المقارنة:')
    print(f'   ماركت (عادي):     مجموع {arr_n.sum():+.2f}% | متوسط {arr_n.mean():+.3f}%')
    print(f'   حد -0.2%:         مجموع {arr_l.sum():+.2f}% | متوسط {arr_l.mean():+.3f}%')
    print(f'   الفرق:            {arr_l.sum() - arr_n.sum():+.2f}%')

print(f'\n✅ تم')
