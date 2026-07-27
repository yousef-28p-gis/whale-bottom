#!/usr/bin/env python3
"""Download 15m OHLCV for all needed symbol-month combos"""
import ccxt, json, os, time
from datetime import datetime, timezone

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED={'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

os.makedirs(CACHE, exist_ok=True)

with open(SIGNALS_FILE) as f: raw=json.load(f)

needed=set()
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction','LONG')!='LONG': continue
    if s.get('volume_usdt',0)<200000: continue
    dt=datetime.fromisoformat(s['dt'])
    if dt.month in (4,5,6) and dt.year==2026:
        needed.add((s['symbol'], dt.strftime('%Y-%m')))

print(f'Need {len(needed)} symbol-month combos')

exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

# Pre-compute date ranges
from calendar import monthrange
import pandas as pd

def get_month_range(mon_str):
    y,m=mon_str.split('-')
    y=int(y); m=int(m)
    start=datetime(y,m,1,tzinfo=timezone.utc)
    # Include a few days before for lookback
    if m==1:
        prev_start=datetime(y-1,12,28,tzinfo=timezone.utc)
    else:
        prev_start=datetime(y,m-1,28,tzinfo=timezone.utc)
    end=datetime(y,m,monthrange(y,m)[1],23,59,tzinfo=timezone.utc)
    return prev_start, end

done=0; skipped=0; failed=0
total=len(needed)

for sym,mon in sorted(needed):
    fpath=f'{CACHE}/{sym}_{mon}.json'
    if os.path.exists(fpath):
        skipped+=1
        continue
    
    prev_start, end = get_month_range(mon)
    since_ms = int(prev_start.timestamp() * 1000)
    
    try:
        candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since_ms, limit=4000)
    except Exception as e:
        failed+=1
        if failed<=5:
            print(f'  FAIL {sym} {mon}: {e}')
        continue
    
    if not candles:
        failed+=1
        continue
    
    data=[{'ts':c[0],'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5]} for c in candles]
    with open(fpath,'w') as f:
        json.dump(data,f)
    
    done+=1
    if done%20==0:
        print(f'  Progress: {done+skipped}/{total} (done={done} skip={skipped} fail={failed})')

print(f'\nDone! Downloaded={done} Skipped={skipped} Failed={failed}')
