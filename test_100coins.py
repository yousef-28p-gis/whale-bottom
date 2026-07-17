#!/usr/bin/env python3
"""Test whale strategy on 100 coins — June 2026. Batch version."""
import ccxt, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})

# Get 100 USDT pairs from Binance
markets = exchange.load_markets()
usdt_pairs = [s for s in markets if s.endswith('/USDT') and markets[s]['active']]
import random
random.seed(42)
test_pairs = random.sample(usdt_pairs, min(100, len(usdt_pairs)))
coins = [p.replace('/USDT','') for p in test_pairs]

print(f'⏳ اختبار {len(coins)} عملة — شهر يونيو 2026...')
print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h')

all_trades = []; coin_stats = []; done=0; skipped=0

for sym in coins:
    try:
        since = exchange.parse8601('2026-06-01T00:00:00Z')
        candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=3000)
        if len(candles) < 500:
            skipped+=1; continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        
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
        
        trades = []
        for i in range(50, len(df)-10):
            row = df.iloc[i]
            if not (row['entry'] and float(row['whale']) >= WHALE_MIN): continue
            if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
            ps = max(0, i-96)
            pb = float(df.iloc[ps]['close'])
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
            trades.append({'pnl':pnl, 'exit':exit_})
        
        if trades:
            wins=sum(1 for t in trades if t['pnl']>0)
            coin_stats.append({'sym':sym,'n':len(trades),'wr':wins/len(trades)*100,'pnl':sum(t['pnl'] for t in trades)})
            all_trades.extend(trades)
    except: skipped+=1
    
    done+=1
    if done%20==0: print(f'  {done}/{len(coins)}... {len(all_trades)} trades', flush=True)

# Summary
print(f'\n{"="*60}')
nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
wr=wins/len(all_trades)*100 if all_trades else 0
exits=defaultdict(int)
for t in all_trades: exits[t['exit']]+=1

print(f'عملات تم اختبارها: {done-skipped} (تخطي: {skipped})')
print(f'إجمالي الصفقات: {len(all_trades)}')
print(f'🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades)-wins}')
print(f'WR: {wr:.1f}%')
print(f'صافي: {sum(nets):+.1f}%')
print(f'🎯TP={exits.get("TP",0)} 🛑SL={exits.get("SL",0)} 🐌TRAIL={exits.get("TRAIL",0)} ⏰TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print(f'متوسط/عملة: {len(all_trades)/(done-skipped):.1f} صفقة')

# Top & worst
coin_stats.sort(key=lambda c: -c['pnl'])
print(f'\n🔝 أفضل ٥:')
for c in coin_stats[:5]:
    print(f'  {c["sym"]}: {c["n"]} صفقة | WR {c["wr"]:.0f}% | صافي {c["pnl"]:+.1f}%')
print(f'\n🔴 أسوأ ٥:')
for c in coin_stats[-5:]:
    print(f'  {c["sym"]}: {c["n"]} صفقة | WR {c["wr"]:.0f}% | صافي {c["pnl"]:+.1f}%')
