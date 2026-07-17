#!/usr/bin/env python3
"""Build cache for April+May 2026 whale sniper signals."""
import json, os, ccxt
from datetime import datetime

SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
CACHE = '/data/trading28/cache/ohlcv'
os.makedirs(CACHE, exist_ok=True)

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED = {'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

with open(SIGNALS_FILE) as f:
    raw = json.load(f)

needed = {}
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction', 'LONG') != 'LONG': continue
    if s.get('volume_usdt', 0) < 200000: continue
    dt = datetime.fromisoformat(s['dt'])
    if dt.year != 2026 or dt.month not in (4, 5): continue
    key = f"{s['symbol']}_{dt.strftime('%Y-%m')}"
    needed[key] = (s['symbol'], dt.strftime('%Y-%m'))

to_fetch = [(sym, mon) for key, (sym, mon) in needed.items() 
            if not os.path.exists(f'{CACHE}/{key}.json')]

print(f'To fetch: {len(to_fetch)}')
if not to_fetch:
    print('All done!')
    exit(0)

exchange = ccxt.binance()
done = 0; fail = 0

for sym, mon in sorted(to_fetch):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    try:
        st = datetime.strptime(mon, '%Y-%m')
        if st.month == 12: end_st = datetime(st.year+1, 1, 1)
        else: end_st = datetime(st.year, st.month+1, 1)
        
        since = int(st.timestamp() * 1000) - 5*24*60*60*1000
        end_ts = int(end_st.timestamp() * 1000) + 7*24*60*60*1000
        max_since = end_ts - 60*24*60*60*1000
        if since < max_since: since = max_since
        
        all_c = []
        while since < end_ts:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=1000)
            if not candles: break
            all_c.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < 1000: break
        
        data = [{'ts': c[0], 'o': c[1], 'h': c[2], 'l': c[3], 'c': c[4], 'v': c[5]} for c in all_c]
        with open(fpath, 'w') as f: json.dump(data, f)
        done += 1
        if done % 50 == 0: print(f'{done}/{len(to_fetch)} ({sym} {mon})')
    except Exception as e:
        fail += 1
        if fail <= 5: print(f'FAIL {sym} {mon}: {e}')

print(f'DONE: {done} fetched, {fail} failed')
