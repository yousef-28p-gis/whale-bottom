#!/usr/bin/env python3 -u
"""تحميل 3 سنوات شمعات 15m لأفضل 10 عملات"""
import ccxt, json, os, time
from datetime import datetime

TOP10 = ['OPN', 'NEIRO', 'CHR', 'MORPHO', 'JTO', 'TOWNS', 'NOM', 'PORTAL', 'MUBARAK', 'CRV']
CACHE_DIR = '/data/trading28/cache/3year'
os.makedirs(CACHE_DIR, exist_ok=True)

# 3 years: July 2023 to July 2026
START = '2023-07-01T00:00:00Z'
END = '2026-07-01T00:00:00Z'

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()

for sym in TOP10:
    fpath = f'{CACHE_DIR}/{sym}_15m.json'
    if os.path.exists(fpath):
        print(f'⏭️ {sym}: موجود مسبقاً', flush=True)
        continue
    
    print(f'📥 {sym}: جاري التحميل...', flush=True)
    since = exchange.parse8601(START)
    end_ts = exchange.parse8601(END)
    
    all_candles = []
    fetch_since = since
    iterations = 0
    while fetch_since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
        except Exception as e:
            print(f'  ⚠️ خطأ: {e}، إعادة محاولة...', flush=True)
            time.sleep(2)
            continue
        
        if not candles:
            break
        
        all_candles.extend(candles)
        iterations += 1
        
        if candles[-1][0] >= end_ts:
            break
        if len(candles) < 1000:
            break
        
        fetch_since = candles[-1][0] + 1
        
        if iterations % 20 == 0:
            elapsed = time.time() - t0
            print(f'  {sym}: {len(all_candles):,} شمعة | {elapsed:.0f}ث', flush=True)
    
    # Save
    with open(fpath, 'w') as f:
        json.dump(all_candles, f)
    
    elapsed = time.time() - t0
    days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400) if len(all_candles) > 1 else 0
    print(f'  ✅ {sym}: {len(all_candles):,} شمعة | {days:.0f} يوم | {elapsed:.0f}ث', flush=True)

total = time.time() - t0
print(f'\n✨ تم تحميل جميع العملات | الوقت: {total:.0f}ث')
