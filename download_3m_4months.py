#!/usr/bin/env python3
"""تحميل 4 شهور بيانات 3m لـ 212 عملة حلال"""
import ccxt, json, os, time
from datetime import datetime, timezone, timedelta

OUT = '/data/trading28/data/3m_4months'
os.makedirs(OUT, exist_ok=True)

with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة")

exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

# 4 months: March 27 - July 27, 2026
end = datetime(2026, 7, 27, 23, 59, tzinfo=timezone.utc)
start = end - timedelta(days=122)
since_ms = int(start.timestamp() * 1000)

total_candles = 0
errors = 0
t0 = time.time()

for i, coin in enumerate(COINS):
    all_candles = []
    current_since = since_ms
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(f'{coin}/USDT', '3m', since=current_since, limit=1000)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ❌ {coin}: {e}")
            break
        
        if not candles:
            break
            
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        
        if last_ts <= current_since or len(candles) < 1000:
            break
        current_since = last_ts + 1
    
    if all_candles:
        data = [{'ts': c[0], 'o': c[1], 'h': c[2], 'l': c[3], 'c': c[4], 'v': c[5]} for c in all_candles]
        fpath = f'{OUT}/{coin}.json'
        with open(fpath, 'w') as f:
            json.dump(data, f)
        total_candles += len(data)
    
    if (i+1) % 20 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (len(COINS) - i - 1)
        sz = sum(os.path.getsize(f'{OUT}/{c}.json') for c in COINS[:i+1] if os.path.exists(f'{OUT}/{c}.json'))
        print(f"  ⏳ {i+1}/{len(COINS)} | {total_candles:,} شمعة | {sz/1024**2:.0f}MB | ⏱️ {elapsed:.0f}s | ETA {eta:.0f}s")

elapsed = time.time() - t0
sz = sum(os.path.getsize(f'{OUT}/{c}.json') for c in COINS if os.path.exists(f'{OUT}/{c}.json'))
print(f"\n✅ تم!")
print(f"   📊 {total_candles:,} شمعة")
print(f"   💾 {sz/1024**2:.0f}MB")
print(f"   🪙 {len(os.listdir(OUT))} عملة")
print(f"   ❌ {errors} أخطاء")
print(f"   ⏱️ {elapsed:.0f}s")
