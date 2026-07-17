#!/usr/bin/env python3
"""Test whale strategy on 10 random coins — 1 month (June 2026)"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20

# 10 diverse coins
COINS = ['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','DOT']

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})

print(f'⏳ جاري جلب بيانات شهر يونيو 2026 لـ {len(COINS)} عملات...')
print(f'⚙️ TP={TP} SL={SL} PL={PL} TRAIL={TRAIL} MH={MH}h WHALE≥{WHALE_MIN}')
print(f'{"="*60}')

all_trades = []

for sym in COINS:
    try:
        # Fetch June 2026
        since = exchange.parse8601('2026-06-01T00:00:00Z')
        candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=3000)
        
        if len(candles) < 500:
            print(f'{sym}: ❌ بيانات غير كافية ({len(candles)})')
            continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Whale indicator
        df = df.copy(); LB=30
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
        
        # Find entries + simulate
        trades = []
        for i in range(50, len(df)-10):
            row = df.iloc[i]
            if not (row['entry'] and float(row['whale']) >= WHALE_MIN): continue
            
            # Single candle
            if i+1 < len(df):
                if float(df.iloc[i+1]['whale']) >= 0.35: continue
            
            # Pump24
            ps = max(0, i-96)
            pb = float(df.iloc[ps]['close'])
            ep = float(row['close'])
            pump24 = (ep-pb)/pb*100 if pb!=0 else 0
            if pump24 >= 0: continue
            
            # Simulate
            tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
            pl_p=ep+(tp_p-ep)*(PL/100)
            pl_trig=False; peak=ep; trail_p=0
            for k in range(i+1, len(df)):
                cur = float(df.iloc[k]['close']); h = (k-i)*0.25
                if h > MH:
                    pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                if cur >= tp_p:
                    pnl=round(TP-COMM,4); exit_='TP'; break
                if cur <= sl_p:
                    pnl=round(-SL-COMM,4); exit_='SL'; break
                if not pl_trig and cur >= pl_p:
                    pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                if pl_trig:
                    if cur > peak:
                        peak=cur; trail_p=cur*(1-TRAIL/100)
                    if cur <= trail_p:
                        pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
            else:
                pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4); exit_='EOD'
            
            trades.append({'dt': row['ts'], 'pnl': pnl, 'exit': exit_, 'whale': round(float(row['whale']),3)})
        
        # Report per coin
        if trades:
            wins = sum(1 for t in trades if t['pnl']>0)
            nets = sum(t['pnl'] for t in trades)
            exits = defaultdict(int)
            for t in trades: exits[t['exit']]+=1
            print(f'\n{sym}: {len(trades)} صفقة | WR {wins/len(trades)*100:.0f}% | صافي {nets:+.1f}%')
            print(f'  TP={exits.get("TP",0)} SL={exits.get("SL",0)} TRAIL={exits.get("TRAIL",0)} TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
            for t in trades:
                icon = '🟢' if t['pnl']>0 else '🔴'
                print(f'  {icon} {t["exit"]} {t["pnl"]:+.2f}% | حوت={t["whale"]}')
            all_trades.extend(trades)
        else:
            print(f'\n{sym}: لا توجد صفقات')
    
    except Exception as e:
        print(f'{sym}: ❌ {e}')

# Final summary
if all_trades:
    wins = sum(1 for t in all_trades if t['pnl']>0)
    nets = sum(t['pnl'] for t in all_trades)
    exits = defaultdict(int)
    for t in all_trades: exits[t['exit']]+=1
    print(f'\n{"="*60}')
    print(f'📊 المجموع — {len(all_trades)} صفقة')
    print(f'   🟢 رابحة: {wins} | 🔴 خاسرة: {len(all_trades)-wins}')
    print(f'   WR: {wins/len(all_trades)*100:.1f}%')
    print(f'   صافي: {nets:+.2f}%')
    print(f'   TP={exits.get("TP",0)} SL={exits.get("SL",0)} TRAIL={exits.get("TRAIL",0)} TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
    
    # Simple compounding 1 pos 100%
    all_trades.sort(key=lambda t: t['dt'])
    cap=1000; peak=1000; max_dd=0; active=None
    for t in all_trades:
        if active and t['dt'] < active[0]: continue
        if active:
            et,amt,ec=active; cap+=amt+ec
        pos=cap; pnl_amt=pos*t['pnl']/100
        active=(t['dt']+pd.Timedelta(hours=MH), pos, pnl_amt)
        eq=cap
        if eq>peak: peak=eq
        dd=(eq-peak)/peak*100
        if dd<max_dd: max_dd=dd
        cap-=pos
    if active: cap+=active[1]+active[2]
    print(f'   💰 $1000 → ${cap:.0f} | سحب {max_dd:.2f}%')
else:
    print('\n⚠️ لا توجد صفقات')
