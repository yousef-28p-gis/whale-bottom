#!/usr/bin/env python3
"""Download via Binance REST API directly — no CCXT overhead"""
import json, os, time, requests
from datetime import datetime, timezone

OUT_DIR = '/data/trading28/data/whale_15m_prev'
os.makedirs(OUT_DIR, exist_ok=True)
BASE = 'https://api.binance.com/api/v3/klines'

with open('/data/trading28/final_bot_config.json') as f:
    configs = json.load(f)
symbols = sorted(set(f"{c['sym']}USDT" for c in configs))
print(f'تحميل {len(symbols)} عملة...', flush=True)

START_TS = int(datetime(2024, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_TS   = int(datetime(2025, 9, 22, tzinfo=timezone.utc).timestamp() * 1000)

total_c, errors = 0, 0
session = requests.Session()

for si, sym in enumerate(symbols):
    coin = sym.replace('USDT','')
    out_path = os.path.join(OUT_DIR, f'{coin}.json')
    
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                ex = json.load(f)
            if ex['ts'] and ex['ts'][-1] >= END_TS - 86400000:
                total_c += len(ex['ts'])
                continue
        except: pass
    
    all_ts, all_o, all_h, all_l, all_c = [], [], [], [], []
    since = START_TS
    
    while since < END_TS:
        try:
            resp = session.get(BASE, params={
                'symbol': sym, 'interval': '15m',
                'startTime': since, 'limit': 1000
            }, timeout=15)
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            # Only keep candles before END_TS (our current data start)
            data_before = [c for c in data if c[0] < END_TS]
            for c in data_before:
                all_ts.append(c[0]); all_o.append(float(c[1]))
                all_h.append(float(c[2])); all_l.append(float(c[3]))
                all_c.append(float(c[4]))
            # If we got data after END_TS, we're done with this period
            if any(c[0] >= END_TS for c in data):
                break
            if not data_before and data[-1][0] >= END_TS:
                break
            since = data[-1][0] + 1
            time.sleep(0.05)
        except Exception as e:
            errors += 1
            if errors <= 3: print(f'  ⚠️ {sym}: {e}', flush=True)
            time.sleep(2)
            break
    
    if len(all_ts) < 100:
        print(f'  [{si+1}/{len(symbols)}] {sym} ❌ {len(all_ts)}c', flush=True)
        continue
    
    with open(out_path, 'w') as f:
        json.dump({'ts': all_ts, 'o': all_o, 'h': all_h, 'l': all_l, 'c': all_c}, f)
    
    days = (all_ts[-1]-all_ts[0])/86400000
    total_c += len(all_ts)
    if (si+1) % 10 == 0:
        print(f'  [{si+1}/{len(symbols)}] {sym} ✓ {len(all_ts)}c ({days:.0f}d)', flush=True)

print(f'\n✅ تم: {total_c:,} شمعة → {OUT_DIR}', flush=True)
