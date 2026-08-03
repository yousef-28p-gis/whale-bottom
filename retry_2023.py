#!/usr/bin/env python3
"""Retry failed 2023 downloads for halal+halal2 coins"""
import json, os, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DIR_2023 = '/data/trading28/data/whale_15m_2023'
START = datetime(2023,9,1,tzinfo=timezone.utc)
END = datetime(2024,9,1,tzinfo=timezone.utc)
START_TS = int(START.timestamp()*1000)
END_TS = int(END.timestamp()*1000)
BASE = 'https://api.binance.com/api/v3/klines'

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = set(d['halal'] + d['halal2'])

existing = set(f.replace('.json','') for f in os.listdir(DIR_2023) if f.endswith('.json'))
missing = sorted(tradeable - existing)
print(f"Retrying {len(missing)} missing coins for 2023...")

def download_2023(coin):
    sym = f"{coin}USDT"
    out_path = os.path.join(DIR_2023, f'{coin}.json')
    all_ts, all_o, all_h, all_l, all_c = [], [], [], [], []
    since = START_TS
    retries = 3
    while since < END_TS and retries > 0:
        try:
            resp = requests.get(BASE, params={
                'symbol': sym, 'interval': '15m',
                'startTime': since, 'limit': 1000
            }, timeout=30)
            data = resp.json()
            if not data or not isinstance(data, list):
                retries -= 1
                time.sleep(2)
                continue
            for c in data:
                if c[0] < END_TS:
                    all_ts.append(c[0]); all_o.append(float(c[1]))
                    all_h.append(float(c[2])); all_l.append(float(c[3]))
                    all_c.append(float(c[4]))
            if any(c[0] >= END_TS for c in data): break
            since = data[-1][0] + 1
            time.sleep(0.02)
            retries = 3
        except:
            retries -= 1
            time.sleep(2)
    
    if len(all_ts) >= 100:
        with open(out_path, 'w') as f:
            json.dump({'ts': all_ts, 'o': all_o, 'h': all_h, 'l': all_l, 'c': all_c}, f)
        return (coin, len(all_ts), 'done')
    return (coin, len(all_ts), 'fail')

done_list, fail_list = [], []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(download_2023, c): c for c in missing}
    for i, f in enumerate(as_completed(futures)):
        coin, n, status = f.result()
        if status == 'done':
            done_list.append(f"{coin} ({n})")
        else:
            fail_list.append(f"{coin} ({n})")
        if (i+1) % 10 == 0:
            print(f'  {i+1}/{len(missing)}  done={len(done_list)}  fail={len(fail_list)}', flush=True)

print(f'\n✅ Downloaded: {len(done_list)}')
for d in done_list:
    print(f'  {d}')
print(f'\n❌ Still failed: {len(fail_list)}')
for f in fail_list:
    print(f'  {f}')

# Final
existing = set(f.replace('.json','') for f in os.listdir(DIR_2023) if f.endswith('.json'))
have = tradeable & existing
print(f'\n2023 tradeable: {len(have)}/212')

# Common check
prev_coins = set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_prev') if f.endswith('.json'))
cur_coins = set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_1y') if f.endswith('.json'))
common = have & prev_coins & cur_coins
print(f'Common all 3: {len(common)}/212')
if len(common) < 212:
    print(f'Missing: {sorted(tradeable - common)}')
