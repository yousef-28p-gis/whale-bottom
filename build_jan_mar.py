#!/usr/bin/env python3
"""Simple cache build — no signal, just try/except"""
import json, ccxt, os, time, sys
from datetime import datetime

CACHE='/data/trading28/cache/ohlcv'
os.makedirs(CACHE, exist_ok=True)
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

with open('/data/trading28/signals_whalesniper_all.json') as f: raw=json.load(f)

needed={}
for s in raw:
    if s['symbol'] in STABLES or s.get('direction')!='LONG' or s.get('volume_usdt',0)<200000: continue
    dt=datetime.fromisoformat(s['dt'])
    if dt.year!=2026 or dt.month not in (1,2,3): continue
    key=f"{s['symbol']}_{dt.strftime('%Y-%m')}"
    if not os.path.exists(f'{CACHE}/{key}.json'):
        needed[key]=(s['symbol'],dt.strftime('%Y-%m'))

print(f"To fetch: {len(needed)}", flush=True)

ex=ccxt.binance({'enableRateLimit':True, 'timeout':20000})
fetched=0; errors=0; t0=time.time()

for key,(sym,mon) in sorted(needed.items()):
    try:
        since=ex.parse8601(f'{mon}-01T00:00:00Z')
        if mon=='2026-03': end=ex.parse8601('2026-04-01T00:00:00Z')
        elif mon=='2026-02': end=ex.parse8601('2026-03-01T00:00:00Z')
        else: end=ex.parse8601('2026-02-01T00:00:00Z')
        
        all_ohlcv=[]
        current=since
        while len(all_ohlcv)<3000:  # max 3000 candles
            candles=ex.fetch_ohlcv(f'{sym}/USDT','15m',since=current,limit=1000)
            if not candles: break
            all_ohlcv.extend(candles)
            last_ts=candles[-1][0]
            if last_ts>=end: break
            if len(candles)<1000: break
            current=last_ts+1
        
        if all_ohlcv:
            out=[{'ts':c[0],'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5]} for c in all_ohlcv]
            with open(f'{CACHE}/{key}.json','w') as f:
                json.dump(out,f)
            fetched+=1
    except Exception as e:
        errors+=1
        if errors<=5: print(f'  ERROR {sym}/{mon}: {e}', flush=True)
    
    if (fetched+errors)%25==0:
        elapsed=time.time()-t0
        print(f'  [{fetched}/{len(needed)}] ok={fetched} err={errors} ({elapsed:.0f}s)', flush=True)

elapsed=time.time()-t0
print(f'\nDone! {fetched} fetched, {errors} errors in {elapsed:.0f}s', flush=True)
