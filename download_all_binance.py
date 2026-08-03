#!/usr/bin/env python3
"""Get ALL USDT pairs from Binance, download all 3 periods"""
import json, os, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DIRS = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'prev': '/data/trading28/data/whale_15m_prev',
    'cur': '/data/trading28/data/whale_15m_1y',
}
PERIODS = {
    '2023': (datetime(2023,9,1,tzinfo=timezone.utc), datetime(2024,9,1,tzinfo=timezone.utc)),
    'prev': (datetime(2024,9,1,tzinfo=timezone.utc), datetime(2025,9,1,tzinfo=timezone.utc)),
    'cur':  (datetime(2025,9,1,tzinfo=timezone.utc), datetime(2026,9,1,tzinfo=timezone.utc)),
}
BASE = 'https://api.binance.com/api/v3/klines'

# Step 1: Get all USDT pairs from Binance
print("Fetching all USDT pairs from Binance...")
info = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=15).json()
all_usdt = sorted([s['symbol'] for s in info['symbols'] 
                   if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'])
print(f"Found {len(all_usdt)} USDT pairs on Binance")

def download_period(sym, period):
    coin = sym.replace('USDT','')
    out_path = os.path.join(DIRS[period], f'{coin}.json')
    start, end = PERIODS[period]
    START_TS = int(start.timestamp()*1000)
    END_TS = int(end.timestamp()*1000)
    
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                ex = json.load(f)
            if ex.get('ts') and ex['ts'][-1] >= END_TS - 86400000:
                return (coin, period, len(ex['ts']), 'skip')
        except: pass
    
    all_ts, all_o, all_h, all_l, all_c = [], [], [], [], []
    since = START_TS
    while since < END_TS:
        try:
            resp = requests.get(BASE, params={
                'symbol': sym, 'interval': '15m',
                'startTime': since, 'limit': 1000
            }, timeout=15)
            data = resp.json()
            if not data or not isinstance(data, list): break
            for c in data:
                if c[0] < END_TS:
                    all_ts.append(c[0]); all_o.append(float(c[1]))
                    all_h.append(float(c[2])); all_l.append(float(c[3]))
                    all_c.append(float(c[4]))
            if any(c[0] >= END_TS for c in data): break
            since = data[-1][0] + 1
            time.sleep(0.01)
        except: break
    
    if len(all_ts) >= 100:
        with open(out_path, 'w') as f:
            json.dump({'ts': all_ts, 'o': all_o, 'h': all_h, 'l': all_l, 'c': all_c}, f)
        return (coin, period, len(all_ts), 'done')
    return (coin, period, 0, 'fail')

# Build tasks for ALL pairs, ALL periods (skip existing)
tasks = []
for sym in all_usdt:
    for period in ['2023','prev','cur']:
        coin = sym.replace('USDT','')
        out_path = os.path.join(DIRS[period], f'{coin}.json')
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    ex = json.load(f)
                _, end = PERIODS[period]
                if ex.get('ts') and ex['ts'][-1] >= int(end.timestamp()*1000) - 86400000:
                    continue
            except: pass
        tasks.append((sym, period))

print(f"Need to download: {len(tasks)} coin-periods")
if not tasks:
    print("Nothing to download!")
else:
    total_done, total_skip, total_fail = 0, 0, 0
    new_2023, new_prev, new_cur = set(), set(), set()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(download_period, s, p): (s,p) for s,p in tasks}
        for i, f in enumerate(as_completed(futures)):
            coin, period, n, status = f.result()
            if status == 'done':
                total_done += n
                if period == '2023': new_2023.add(coin)
                elif period == 'prev': new_prev.add(coin)
                else: new_cur.add(coin)
            elif status == 'skip': total_skip += 1
            else: total_fail += 1
            if (i+1) % 100 == 0:
                print(f'  {i+1}/{len(tasks)} done={total_done} skip={total_skip} fail={total_fail}', flush=True)

    print(f'\nDone: {total_done} candles, {total_skip} skipped, {total_fail} failed')
    print(f'New 2023: {len(new_2023)} | New prev: {len(new_prev)} | New cur: {len(new_cur)}')

# Final
print("\n=== FINAL ===")
coins = {}
for period in ['2023','prev','cur']:
    s = set(f.replace('.json','') for f in os.listdir(DIRS[period]) if f.endswith('.json'))
    coins[period] = s
    print(f"{period}: {len(s)} coins")

common = coins['2023'] & coins['prev'] & coins['cur']
print(f"Common (all 3 years): {len(common)} coins  —  target: 212")
