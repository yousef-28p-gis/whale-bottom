#!/usr/bin/env python3
"""Build OHLCV cache for Whale Sniper LONG signals — June+July 2026 only"""
import json, os, ccxt
from datetime import datetime, timezone

SIGNALS_FILE = '/data/trading28/signals_ws_single.json'
CACHE = '/data/trading28/cache/ohlcv'
os.makedirs(CACHE, exist_ok=True)

with open(SIGNALS_FILE) as f:
    signals = json.load(f)

# Filter LONG + non-stable
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
longs = [s for s in signals if s['direction'] == 'LONG' and s['symbol'] not in STABLES]

# Collect needed symbol-months
needed = {}
for s in longs:
    dt = datetime.fromisoformat(s['dt'])
    mon = dt.strftime('%Y-%m')
    key = f"{s['symbol']}_{mon}"
    needed[key] = (s['symbol'], mon)

# Filter already cached
to_fetch = {}
for key, (sym, mon) in needed.items():
    fpath = f'{CACHE}/{key}.json'
    if not os.path.exists(fpath):
        to_fetch[key] = (sym, mon)

print(f'Total needed: {len(needed)}, already cached: {len(needed)-len(to_fetch)}, to fetch: {len(to_fetch)}')

if not to_fetch:
    print('All cached!')
    exit(0)

exchange = ccxt.binance()
done = 0; fail = 0

for key, (sym, mon) in sorted(to_fetch.items()):
    fpath = f'{CACHE}/{key}.json'
    try:
        st = datetime.strptime(mon, '%Y-%m')
        if st.month == 12:
            end_st = datetime(st.year+1, 1, 1)
        else:
            end_st = datetime(st.year, st.month+1, 1)
        
        since = int(st.timestamp() * 1000) - 5*24*60*60*1000
        end_ts = int(end_st.timestamp() * 1000) + 7*24*60*60*1000
        
        # Cap at 60 days
        max_since = end_ts - 60*24*60*60*1000
        if since < max_since:
            since = max_since
        
        all_candles = []
        while since < end_ts:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < 1000:
                break
        
        # Convert to list of dicts
        data = [{'ts': c[0], 'o': c[1], 'h': c[2], 'l': c[3], 'c': c[4], 'v': c[5]} for c in all_candles]
        
        with open(fpath, 'w') as f:
            json.dump(data, f)
        
        done += 1
        if done % 20 == 0:
            print(f'  {done}/{len(to_fetch)} done ({sym} {mon}), {fail} failed')
    
    except Exception as e:
        fail += 1
        print(f'  FAIL {sym} {mon}: {e}')

print(f'\nDone: {done} fetched, {fail} failed')

# Final cache count
import glob
cached = glob.glob(f'{CACHE}/*.json')
print(f'Total cache files: {len(cached)}')
