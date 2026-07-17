#!/usr/bin/env python3
"""REKTbinance top coins → whale entry → live simulation"""
import re, subprocess, ccxt, numpy as np, pandas as pd
from html import unescape
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.40; COMM=0.20

# ── Scrape ──
print('⏳ سحب REKTbinance...')
all_buys = []
target_id = None
for page in range(20):
    url = 'https://t.me/s/REKTbinance' if page==0 else f'https://t.me/s/REKTbinance?embed=1&before={target_id}'
    result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, timeout=30)
    html = unescape(result.stdout)
    msgs = re.findall(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
    for msg in msgs:
        text = re.sub(r'<[^>]+>', ' ', msg)
        text = re.sub(r'\s+', ' ', text).strip()
        for m in re.finditer(r'(\w+USDT)\s+BUY\s+([\d,.]+)\s+@\s+([\d.]+)', text):
            all_buys.append({'symbol': m.group(1), 'qty': float(m.group(2).replace(',','')), 'price': float(m.group(3))})
    ids = re.findall(r'data-post="REKTbinance/(\d+)"', html)
    if not ids: break
    target_id = min(int(x) for x in ids)-1

sym_counts = Counter(b['symbol'] for b in all_buys)
top10 = sym_counts.most_common(10)

# ── Fetch & Simulate ──
print(f'\n{"="*65}')
print(f'🐋 محاكاة صفقات — REKT BUY تصفية + تأكيد حوت')
print(f'   TP={TP}% SL={SL}% PL={PL}% TRAIL={TRAIL}% MH={MH}h')
print(f'{"="*65}')

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
all_results = []

for sym, liq_count in top10:
    try:
        since = exchange.parse8601((datetime.now(timezone.utc)-timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'))
        candles = exchange.fetch_ohlcv(f'{sym}', '15m', since=since, limit=500)
        if len(candles) < 200: continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # Whale
        LB=30; df=df.copy()
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
        
        # Find whale confirmations in last 72h
        cutoff = pd.Timestamp(datetime.now(timezone.utc)-timedelta(hours=72)).tz_localize(None)
        recent = df[df['ts'] > cutoff].reset_index(drop=True)
        
        trades = []
        for i in range(len(recent)):
            row = recent.iloc[i]
            if not (row['entry'] and float(row['whale']) >= WHALE_MIN): continue
            
            # Single candle check
            if i+1 < len(recent):
                wn = float(recent.iloc[i+1]['whale'])
                if wn >= 0.35: continue
            
            # Pump24 check
            orig_idx = recent.index[i] if isinstance(recent.index, pd.RangeIndex) else i
            ps = max(0, orig_idx-96)
            if ps < len(df):
                pb = float(df.iloc[ps]['close'])
                ep = float(row['close'])
                pump24 = (ep-pb)/pb*100 if pb!=0 else 0
                if pump24 >= 0: continue
            
            # Simulate
            ep = float(row['close'])
            tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
            pl_p=ep+(tp_p-ep)*(PL/100)
            pl_trig=False; peak=ep; trail_p=0; pnl=None; exit_='?'
            
            for k in range(i+1, len(recent)):
                cur = float(recent.iloc[k]['close']); h = (k-i)*0.25
                if h > MH:
                    pnl = round((cur-ep)/ep*100-COMM,4); exit_='⏰ وقت'; break
                if cur >= tp_p:
                    pnl = round(TP-COMM,4); exit_='🎯 هدف'; break
                if cur <= sl_p:
                    pnl = round(-SL-COMM,4); exit_='🛑 ستوب'; break
                if not pl_trig and cur >= pl_p:
                    pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                if pl_trig:
                    if cur > peak:
                        peak=cur; trail_p=cur*(1-TRAIL/100)
                    if cur <= trail_p:
                        pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='🐌 تريل'; break
            else:
                cur = float(recent.iloc[-1]['close'])
                pnl=round((cur-ep)/ep*100-COMM,4); exit_='📊 مفتوح'
            
            entry_dt = row['ts']
            trades.append({'dt': entry_dt, 'pnl': pnl, 'exit': exit_, 'ep': round(ep,6), 'wv': round(float(row['whale']),3)})
        
        # Print results
        if trades:
            wins = sum(1 for t in trades if t['pnl']>0)
            print(f'\n{sym} ({liq_count} تصفية):')
            for t in trades:
                icon = '🟢' if t['pnl']>0 else '🔴'
                print(f'  {icon} {str(t["dt"])[:16]} | دخول={t["ep"]} | {t["exit"]} {t["pnl"]:+.2f}% | حوت={t["wv"]}')
            all_results.extend(trades)
        else:
            print(f'\n{sym} ({liq_count} تصفية): لا توجد صفقات')
            
    except Exception as e:
        print(f'\n{sym}: ❌ {e}')

# ── Summary ──
if all_results:
    wins = sum(1 for t in all_results if t['pnl']>0)
    total = len(all_results)
    nets = sum(t['pnl'] for t in all_results)
    print(f'\n{"="*65}')
    print(f'📊 ملخص — {total} صفقة')
    print(f'   🟢 رابحة: {wins} | 🔴 خاسرة: {total-wins}')
    print(f'   WR: {wins/total*100:.1f}%')
    print(f'   صافي: {nets:+.2f}%')
else:
    print('\n⚠️ لا توجد صفقات')
