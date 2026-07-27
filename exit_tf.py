#!/usr/bin/env python3
"""مقارنة خروج 15د vs 1س vs 4س — كتابة لملف"""
import json, os, numpy as np, pandas as pd, sys
from collections import defaultdict

CACHE_DIR = '/data/trading28/data/5year_halal'
OUT_FILE = '/data/trading28/exit_tf_results.txt'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}
COINS = ['ADA','ETH','SOL','DOGE','AVAX','LINK','DOT','ATOM','GRT','SAND']

def log(msg):
    print(msg, flush=True)
    with open(OUT_FILE, 'a') as f:
        f.write(msg + '\n')

with open(OUT_FILE, 'w') as f:
    f.write('')

results = {}

for tf_name, tf_candles in [('15m', 1), ('1h', 4), ('4h', 16)]:
    log(f'⏳ بدء {tf_name}...')
    all_pnl = []
    exit_counts = defaultdict(int)
    
    for sym in COINS:
        fname = f'{sym}_15m.json'
        fp = f'{CACHE_DIR}/{fname}'
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
            
            tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
            pl_p=ep+(tp_p-ep)*(PL/100)
            pl_trig=False; peak=ep; trail_p=0
            pnl = None
            exit_ = 'EOD'
            k = i + tf_candles
            while k < len(df):
                cur=float(df.iloc[k]['close']); h=(k-i)*0.25
                if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
                if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
                if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                if pl_trig:
                    if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                    if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
                k += tf_candles
            if pnl is None:
                pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
            
            all_pnl.append(pnl)
            exit_counts[exit_] += 1
        
        log(f'  {sym}: {sum(1 for x in all_pnl[-10:] if True)} trades')
    
    arr = np.array(all_pnl)
    wins = (arr > 0).sum()
    results[tf_name] = {'trades': len(all_pnl), 'wins': int(wins), 'sum': float(arr.sum()), 'mean': float(arr.mean()), 'exits': dict(exit_counts)}
    log(f'✅ {tf_name}: {len(all_pnl)}ت | WR {wins/len(all_pnl)*100:.1f}% | مجموع {arr.sum():+.1f}%')

log('\n' + '='*60)
log('📊 مقارنة الخروج حسب الإطار الزمني')
log('='*60)
for tf in ['15m', '1h', '4h']:
    r = results[tf]
    wr = r['wins']/r['trades']*100 if r['trades'] > 0 else 0
    log(f'\n⏱️  خروج {tf}:')
    log(f'   صفقات: {r["trades"]} | 🟢 {r["wins"]} | WR {wr:.1f}%')
    log(f'   مجموع: {r["sum"]:+.1f}% | متوسط: {r["mean"]:+.3f}%')
    log(f'   مخارج: {r["exits"]}')

log('\n✅ تم')
