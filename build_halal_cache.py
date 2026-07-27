#!/usr/bin/env python3 -u
"""
بناء كاش 5 سنوات لعملات الحلال فقط + تشغيل الباك تيست مباشرة بعدها
"""
import ccxt, json, os, sys, time
from datetime import datetime

# Load halal coins
with open('/data/trading28/config/shariah_coins.json') as f:
    data = json.load(f)
HALAL = data['halal']

OLD_CACHE = '/data/trading28/cache/5year'
NEW_CACHE = '/data/trading28/data/5year_halal'
START = '2021-07-01T00:00:00Z'

os.makedirs(NEW_CACHE, exist_ok=True)

# Symlink existing coins
linked = 0
for c in HALAL:
    old_path = f'{OLD_CACHE}/{c}_15m.json'
    new_path = f'{NEW_CACHE}/{c}_15m.json'
    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.symlink(os.path.abspath(old_path), new_path)
        linked += 1

# Count already existing (including symlinks)
existing = sum(1 for c in HALAL if os.path.exists(f'{NEW_CACHE}/{c}_15m.json'))
to_download = [c for c in HALAL if not os.path.exists(f'{NEW_CACHE}/{c}_15m.json')]

print(f'✅ حلال: {len(HALAL)} | موجود: {existing} (منها {linked} رابط) | تنزيل: {len(to_download)}')
print(f'⏱️  الوقت المتوقع: ~{len(to_download)*50//60} دقيقة')
print()

if not to_download:
    print('كل العملات موجودة — لا حاجة للتنزيل!')
    sys.exit(0)

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()
errors = []

for idx, sym in enumerate(to_download):
    fpath = f'{NEW_CACHE}/{sym}_15m.json'
    print(f'📥 {idx+1}/{len(to_download)} {sym:<10}', end='', flush=True)
    
    since = exchange.parse8601(START)
    end_ts = int(time.time() * 1000)
    all_candles = []
    fetch_since = since
    
    try:
        while fetch_since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
            except Exception:
                time.sleep(2)
                continue
            
            if not candles:
                break
            all_candles.extend(candles)
            if candles[-1][0] >= end_ts or len(candles) < 1000:
                break
            fetch_since = candles[-1][0] + 1
        
        if all_candles:
            with open(fpath, 'w') as f:
                json.dump(all_candles, f)
            days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400)
            size_mb = os.path.getsize(fpath) / 1_000_000
            elapsed = time.time() - t0
            print(f' ✅ {len(all_candles):,}ش ({days:.0f}يوم, {size_mb:.0f}MB) | الكلي: {elapsed/60:.0f}د', flush=True)
        else:
            print(f' ⚠️  لا بيانات', flush=True)
            errors.append(sym)
    except Exception as e:
        print(f' ❌ {e}', flush=True)
        errors.append(sym)
        continue

total = time.time() - t0
downloaded = len(to_download) - len(errors)
print(f'\n✨ تم: {downloaded}/{len(to_download)} عملة | الوقت: {total/60:.0f} دقيقة')
if errors:
    print(f'⚠️  فشل: {errors}')
print(f'\n📦 الكاش جاهز في: {NEW_CACHE}')
