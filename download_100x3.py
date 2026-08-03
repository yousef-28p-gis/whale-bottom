#!/usr/bin/env python3
"""Download 100 coins × 3 years — skip existing"""
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
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

with open('/data/trading28/top100_coins.json') as f:
    symbols = json.load(f)
print(f"Target: {len(symbols)} coins × 3 years")
BASE = 'https://api.binance.com/api/v3/klines'

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

# Build task list
tasks = []
for sym in symbols:
    for period in ['2023','prev','cur']:
        coin = sym.replace('USDT','')
        if not os.path.exists(os.path.join(DIRS[period], f'{coin}.json')):
            tasks.append((sym, period))

print(f"Need to download: {len(tasks)} coin-periods (out of {len(symbols)*3})")

if not tasks:
    print("All data already exists!")
else:
    total_done, total_skip, total_fail = 0, 0, 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(download_period, s, p): (s,p) for s,p in tasks}
        for i, f in enumerate(as_completed(futures)):
            coin, period, n, status = f.result()
            if status == 'done': total_done += n
            elif status == 'skip': total_skip += 1
            else: total_fail += 1
            if (i+1) % 30 == 0:
                print(f'  {i+1}/{len(tasks)} done={total_done} skip={total_skip} fail={total_fail}', flush=True)
    
    print(f'\nDone: {total_done} candles, {total_skip} skipped, {total_fail} failed', flush=True)

# Verify
print("\nFinal counts:")
for period in ['2023','prev','cur']:
    files = [f for f in os.listdir(DIRS[period]) if f.endswith('.json') and f!='_manifest.json']
    print(f"  {period}: {len(files)} coins")
