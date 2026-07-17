#!/usr/bin/env python3 -u
"""تحميل 5 سنوات شمعات 15m لـ 50 عملة منوعة"""
import ccxt, json, os, time
from datetime import datetime

COINS = [
    'BTC','ETH','BNB','SOL','ADA','AVAX','NEAR','DOT','ATOM','FTM',
    'INJ','KAVA','XTZ','ALGO','EGLD','HBAR','FLOW','DASH','ZIL',
    'MATIC','UNI','AAVE','MKR','CRV','COMP','SUSHI','SNX','YFI','1INCH','RUNE',
    'FET','OCEAN','LINK','BAND','TRB','GRT',
    'AXS','SAND','MANA','THETA','CHZ','ENJ',
    'DOGE','SHIB','FIL','AR','LPT','STORJ','XMR','ZEC'
]

CACHE_DIR = '/data/trading28/cache/5year'
os.makedirs(CACHE_DIR, exist_ok=True)

START = '2021-07-01T00:00:00Z'
exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()

for idx, sym in enumerate(COINS):
    fpath = f'{CACHE_DIR}/{sym}_15m.json'
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1_000_000
        print(f'⏭️ {idx+1}/50 {sym:<8} موجود ({size_mb:.0f}MB)', flush=True)
        continue
    
    print(f'📥 {idx+1}/50 {sym:<8} جاري...', flush=True)
    since = exchange.parse8601(START)
    end_ts = int(time.time() * 1000)
    
    all_candles = []
    fetch_since = since
    iterations = 0
    while fetch_since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
        except:
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
        
        if iterations % 30 == 0:
            elapsed = time.time() - t0
            days = (fetch_since - since) / (1000*86400) if fetch_since > since else 0
            print(f'  {sym}: {len(all_candles):,} شمعة | {days:.0f} يوم | {elapsed:.0f}ث', flush=True)
    
    if all_candles:
        with open(fpath, 'w') as f:
            json.dump(all_candles, f)
    
    elapsed = time.time() - t0
    days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400) if len(all_candles) > 1 else 0
    size_mb = os.path.getsize(fpath) / 1_000_000
    print(f'  ✅ {sym}: {len(all_candles):,} شمعة | {days:.0f} يوم | {size_mb:.0f}MB | {elapsed:.0f}ث', flush=True)

total = time.time() - t0
print(f'\n✨ تم تحميل {len(COINS)} عملة | الوقت: {total/60:.0f} دقيقة')
