#!/usr/bin/env python3 -u
"""بناء كاش 5 سنوات للعملات المفقودة (32 عملة حلال)"""
import ccxt, json, os, time, sys

COINS = [
    '1INCH','ADA','ALGO','AR','ATOM','AVAX','AXS','CHZ','DOGE',
    'DOT','EGLD','ETH','FET','FIL','GRT','HBAR','KAVA','LINK','LPT',
    'MANA','NEAR','SAND','SHIB','SOL','STORJ','SUSHI','THETA','TRB',
    'UNI','XTZ','YFI','ZIL'
]

CACHE_DIR = '/data/trading28/data/5year_halal'
os.makedirs(CACHE_DIR, exist_ok=True)

START = '2021-07-01T00:00:00Z'
exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 30000})
t0 = time.time()
built = 0
skipped = 0
failed = 0

for idx, sym in enumerate(COINS):
    fpath = f'{CACHE_DIR}/{sym}_15m.json'
    
    # Remove broken symlink if exists
    if os.path.islink(fpath) and not os.path.exists(fpath):
        os.unlink(fpath)
        print(f'🧹 {idx+1}/32 {sym:<8} رابط مكسور ← حذف', flush=True)
    
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1_000_000
        print(f'⏭️ {idx+1}/32 {sym:<8} موجود ({size_mb:.0f}MB)', flush=True)
        skipped += 1
        continue
    
    print(f'📥 {idx+1}/32 {sym:<8} جاري...', flush=True)
    since = exchange.parse8601(START)
    end_ts = int(time.time() * 1000)
    
    all_candles = []
    fetch_since = since
    iterations = 0
    
    try:
        while fetch_since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
            except Exception as e:
                print(f'  ⚠️ خطأ API: {e} | انتظار 5ث...', flush=True)
                time.sleep(5)
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
                days = (fetch_since - since) / (1000*86400) if fetch_since > since else 0
                print(f'  {sym}: {len(all_candles):,} شمعة | {days:.0f}يوم | {elapsed:.0f}ث', flush=True)
    
        if all_candles:
            with open(fpath, 'w') as f:
                json.dump(all_candles, f)
            built += 1
            
            elapsed = time.time() - t0
            days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400) if len(all_candles) > 1 else 0
            size_mb = os.path.getsize(fpath) / 1_000_000
            print(f'  ✅ {sym}: {len(all_candles):,} شمعة | {days:.0f}يوم | {size_mb:.0f}MB | {elapsed:.0f}ث', flush=True)
        else:
            failed += 1
            print(f'  ❌ {sym}: لا بيانات', flush=True)
            
    except KeyboardInterrupt:
        print(f'\n⏹️ توقف عند {sym}', flush=True)
        break
    except Exception as e:
        failed += 1
        print(f'  ❌ {sym}: استثناء - {e}', flush=True)

total = time.time() - t0
print(f'\n✨ تم: {built} بنيت | {skipped} متخطية | {failed} فشلت | الوقت: {total/60:.0f}د', flush=True)
