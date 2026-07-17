#!/usr/bin/env python3 -u
"""تنزيل كاش 5 سنوات لعملات المشبوهة القابلة للاختبار"""
import ccxt, json, os, sys, time

TESTABLE = [
    # DEX
    'UNI','SUSHI','CAKE','DYDX','1INCH','COW','DODO','JOE','JUP','ORCA','QUICK','RAY','VELODROME',
    # Fan
    'ACM','ALPINE','ASR','ATM','BAR','CITY','JUV','LAZIO','OG','PORTO','PSG','SANTOS',
    # Gaming
    'AGLD','ALICE','ANIME','AXS','BIGTIME','CATI','ENJ','GALA','HMSTR','ILV','MAGIC','MANA','NOT','PIXEL','SAND','TLM','YGG',
    # Privacy
    'DASH','PIVX','SCRT','XVG','ZEC',
]

OLD_CACHE = '/data/trading28/cache/5year'
HALAL_CACHE = '/data/trading28/cache/5year_halal'
NEW_CACHE = '/data/trading28/cache/5year_halal'
START = '2021-07-01T00:00:00Z'

# Symlink existing
linked = 0
for src_dir in [OLD_CACHE, HALAL_CACHE]:
    for c in TESTABLE:
        dst = f'{NEW_CACHE}/{c}_15m.json'
        if os.path.exists(dst):
            continue
        src = f'{src_dir}/{c}_15m.json'
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)
            linked += 1

to_download = [c for c in TESTABLE if not os.path.exists(f'{NEW_CACHE}/{c}_15m.json')]
print(f'✅ رابط: {linked} | ⬇️ تنزيل: {len(to_download)}')
print()

if not to_download:
    print('كلها موجودة!')
    sys.exit(0)

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()

for idx, sym in enumerate(to_download):
    fpath = f'{NEW_CACHE}/{sym}_15m.json'
    print(f'📥 {idx+1}/{len(to_download)} {sym:<12}', end='', flush=True)
    
    since = exchange.parse8601(START)
    end_ts = int(time.time() * 1000)
    all_candles = []
    fetch_since = since
    
    try:
        while fetch_since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
            except:
                time.sleep(2)
                continue
            if not candles: break
            all_candles.extend(candles)
            if candles[-1][0] >= end_ts or len(candles) < 1000: break
            fetch_since = candles[-1][0] + 1
        
        if all_candles:
            with open(fpath, 'w') as f:
                json.dump(all_candles, f)
            days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400)
            elapsed = time.time() - t0
            print(f' ✅ {len(all_candles):,}ش ({days:.0f}يوم) | {elapsed/60:.0f}د', flush=True)
        else:
            print(f' ⚠️  لا بيانات', flush=True)
    except Exception as e:
        print(f' ❌ {e}', flush=True)

total = time.time() - t0
final = sum(1 for c in TESTABLE if os.path.exists(f'{NEW_CACHE}/{c}_15m.json'))
print(f'\n✨ جاهز: {final}/{len(TESTABLE)} عملة | الوقت: {total/60:.0f} دقيقة')
