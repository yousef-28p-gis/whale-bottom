#!/usr/bin/env python3 -u
"""تنزيل كاش 5 سنوات — ميم + جديدة غير محكمة"""
import ccxt, json, os, time

MEME = ['DOGE','SHIB','PEPE','FLOKI','WIF','BONK','BOME','MEME','TURBO','NEIRO','PNUT','PENGU','1MBABYDOGE','DOGS','TST','BANANAS31','BANANA','BROCCOLI714','TURTLE','1000CHEEMS','1000CAT','1000SATS','MUBARAK','MUB','KAT','CHIP','FOGO','TUT','MMT','GIGGLE','BABY','DOLO','PUMP','SLP','WIN','XPL']

NEW = ['ACX','BMT','EDEN','EPIC','ERA','FORM','GPS','GRAM','HAEDAL','HOME','HUMA','IQ','LAYER','LUMIA','MANTA','MANTRA','MITO','PLUME','RESOLV','SCR','SHELL','SKY','SPK','STO','SYRUP','THE','U','YB','ZKC','AT','GUN','HEMI','KAIA','KAITO','NIGHT','NIL','NXPC','OPG','PROVE','SOPH','TREE','W','ZBT','A','AVNT','AWE','BANK','BARD','C','F','G','GENIUS','KGST','NOM','OPN','RE','S','WCT']

TO_DOWNLOAD = MEME + NEW

CACHE_DIR = '/data/trading28/data/5year_halal'
START = '2021-07-01T00:00:00Z'

# Link existing
halal_cache = {f.replace('_15m.json','') for f in os.listdir(CACHE_DIR) if f.endswith('.json')}
old_cache = {f.replace('_15m.json','') for f in os.listdir('/data/trading28/cache/5year') if f.endswith('.json')}

for c in TO_DOWNLOAD:
    dst = f'{CACHE_DIR}/{c}_15m.json'
    if os.path.exists(dst): continue
    for sd in [CACHE_DIR, '/data/trading28/cache/5year']:
        src = f'{sd}/{c}_15m.json'
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)
            break

to_dl = [c for c in TO_DOWNLOAD if not os.path.exists(f'{CACHE_DIR}/{c}_15m.json')]
print(f'⬇️ تنزيل: {len(to_dl)}/{len(TO_DOWNLOAD)}')
if not to_dl:
    print('كلها موجودة!')
    exit()
print()

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()
errors = []

for idx, sym in enumerate(to_dl):
    fpath = f'{CACHE_DIR}/{sym}_15m.json'
    print(f'📥 {idx+1}/{len(to_dl)} {sym:<14}', end='', flush=True)
    
    since = exchange.parse8601(START)
    end_ts = int(time.time() * 1000)
    all_candles = []
    fetch_since = since
    
    try:
        while fetch_since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=fetch_since, limit=1000)
            except: time.sleep(2); continue
            if not candles: break
            all_candles.extend(candles)
            if candles[-1][0] >= end_ts or len(candles) < 1000: break
            fetch_since = candles[-1][0] + 1
        
        if all_candles:
            with open(fpath, 'w') as f: json.dump(all_candles, f)
            days = (all_candles[-1][0] - all_candles[0][0]) / (1000*86400)
            elapsed = time.time() - t0
            print(f' ✅ {len(all_candles):,}ش ({days:.0f}يوم) | {elapsed/60:.0f}د', flush=True)
        else:
            print(f' ⚠️  لا بيانات', flush=True)
            errors.append(sym)
    except Exception as e:
        print(f' ❌ {e}', flush=True)
        errors.append(sym)

total = time.time() - t0
done = sum(1 for c in TO_DOWNLOAD if os.path.exists(f'{CACHE_DIR}/{c}_15m.json'))
print(f'\n✨ {done}/{len(TO_DOWNLOAD)} | {total/60:.0f}د | ❌ {len(errors)}')
