#!/usr/bin/env python3
"""Test whale strategy on ALL Binance USDT pairs — June 2026"""
import ccxt, numpy as np, pandas as pd
from collections import defaultdict
import time

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
markets = exchange.load_markets()
coins = [s.replace('/USDT','') for s in markets if s.endswith('/USDT') and markets[s]['active']]

print(f'⏳ اختبار {len(coins)} زوج — يونيو 2026')
print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN}')
t0=time.time()

all_trades=[]; done=0; skipped=0; errors=0

for sym in coins:
    try:
        since = exchange.parse8601('2026-06-01T00:00:00Z')
        candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=3000)
        if len(candles) < 500:
            skipped+=1; done+=1; continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        
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
        
        # Find + simulate
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
            all_trades.append({'dt': row['ts'], 'pnl': pnl, 'exit': exit_, 'sym': sym})
    except:
        errors+=1
    
    done+=1
    if done%50==0:
        elapsed = time.time()-t0
        print(f'  {done}/{len(coins)} | {len(all_trades)} صفقة | {elapsed:.0f}ث', flush=True)

# ── Results ──
elapsed = time.time()-t0
nets=[t['pnl'] for t in all_trades]
wins=sum(1 for n in nets if n>0)
wr=wins/len(all_trades)*100 if all_trades else 0
exits=defaultdict(int)
for t in all_trades: exits[t['exit']]+=1

print(f'\n{"="*55}')
print(f'📊 {len(coins)} زوج — شهر يونيو 2026')
print(f'{"="*55}')
print(f'تم: {done-skipped-errors} | تخطي: {skipped} | أخطاء: {errors}')
print(f'صفقات: {len(all_trades)}')
print(f'🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades)-wins}')
print(f'WR: {wr:.1f}%')
print(f'صافي: {sum(nets):+.1f}%')
print(f'🎯TP={exits.get("TP",0)} 🛑SL={exits.get("SL",0)} 🐌TRAIL={exits.get("TRAIL",0)} ⏰TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
print(f'متوسط/عملة: {len(all_trades)/(done-skipped-errors):.1f}')
print(f'الوقت: {elapsed:.0f} ثانية')

# Portfolio 1x100%
all_trades.sort(key=lambda t: t['dt'])
cap=1000; peak=1000; max_dd=0; active=None; taken=0
for t in all_trades:
    if active and t['dt'] < active[0]: continue
    if active:
        et,amt,ec=active; cap+=amt+ec
    taken+=1
    pos=cap; pnl_amt=pos*t['pnl']/100
    active=(t['dt']+pd.Timedelta(hours=MH), pos, pnl_amt)
    eq=cap
    if eq>peak: peak=eq
    dd=(eq-peak)/peak*100
    if dd<max_dd: max_dd=dd
    cap-=pos
if active: cap+=active[1]+active[2]
print(f'💰 محفظة: $1000 → ${cap:.0f} | سحب: {max_dd:.2f}% | منفذة: {taken}')
