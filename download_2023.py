#!/usr/bin/env python3
"""Download Sep 2023 - Sep 2024 (3rd year of data)"""
import json, os, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR = '/data/trading28/data/whale_15m_2023'
os.makedirs(OUT_DIR, exist_ok=True)
BASE = 'https://api.binance.com/api/v3/klines'
START_TS = int(datetime(2023, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_TS   = int(datetime(2024, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)

with open('/data/trading28/final_bot_config.json') as f:
    symbols = sorted(set(f"{c['sym']}USDT" for c in json.load(f)))

def download_coin(sym):
    coin = sym.replace('USDT', '')
    out_path = os.path.join(OUT_DIR, f'{coin}.json')
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                ex = json.load(f)
            if ex.get('ts') and ex['ts'][-1] >= END_TS - 86400000:
                return coin, len(ex['ts']), True
        except: pass
    
    all_ts, all_o, all_h, all_l, all_c = [], [], [], [], []
    since = START_TS
    while since < END_TS:
        try:
            resp = requests.get(BASE, params={
                'symbol': sym, 'interval': '15m',
                'startTime': since, 'limit': 1000
            }, timeout=10)
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            data_before = [c for c in data if c[0] < END_TS]
            for c in data_before:
                all_ts.append(c[0]); all_o.append(float(c[1]))
                all_h.append(float(c[2])); all_l.append(float(c[3]))
                all_c.append(float(c[4]))
            if any(c[0] >= END_TS for c in data):
                break
            since = data[-1][0] + 1
            time.sleep(0.02)
        except:
            break
    
    if len(all_ts) >= 100:
        with open(out_path, 'w') as f:
            json.dump({'ts': all_ts, 'o': all_o, 'h': all_h, 'l': all_l, 'c': all_c}, f)
    return coin, len(all_ts), False

print(f'Downloading {len(symbols)} coins (10 threads) — Sep2023→Sep2024...', flush=True)
total_c, skipped, failed = 0, 0, 0

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(download_coin, s): s for s in symbols}
    for i, f in enumerate(as_completed(futures)):
        coin, n, was_skip = f.result()
        if was_skip:
            skipped += 1
        elif n >= 100:
            total_c += n
        else:
            failed += 1
        if (i+1) % 15 == 0:
            print(f'  {i+1}/{len(symbols)} total_c={total_c} skipped={skipped} failed={failed}', flush=True)

print(f'\nDone: {total_c} candles, {skipped} skipped, {failed} no-data', flush=True)
