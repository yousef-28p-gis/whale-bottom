#!/usr/bin/env python3 -u
"""تنزيل كاش — رمادي قديم 104 عملة"""
import ccxt, json, os, time

OLD_GREY = ['ACE','AIXBT','ALT','APE','ARPA','ASTER','AUDIO','AXL','BAND','BB','BEAMX','BICO','BLUR','BNT','C98','CELO','CELR','CETUS','CFG','CFX','CGPT','CHZ','COTI','CTK','CTSI','CVX','CYBER','DCR','DEXE','DUSK','EGLD','ENS','ETHFI','FLOW','GLMR','GMT','GNO','GTC','HEI','HFT','ICP','ID','JST','KAVA','KNC','KSM','MASK','MAV','MET','METIS','MINA','MOVR','MTL','NEWT','OGN','ONDO','ONE','ONT','OP','ORDI','OSMO','PAXG','PEOPLE','POL','POLYX','PORTAL','PYR','QI','QNT','RAD','RARE','REZ','RIF','RLC','RONIN','ROSE','RSR','RUNE','RVN','SEI','SFP','SKL','SPELL','SUPER','SYN','T','TNSR','TWT','UMA','VANA','VANRY','VIC','VIRTUAL','WAXP','WOO','XAI','XAUT','XVS','YFI','ZAMA','ZEN','ZK','ZKP','ZRX']

CACHE_DIR = '/data/trading28/data/5year_halal'
START = '2021-07-01T00:00:00Z'

# Link existing
for c in OLD_GREY:
    dst = f'{CACHE_DIR}/{c}_15m.json'
    if os.path.exists(dst): continue
    for sd in [CACHE_DIR, '/data/trading28/cache/5year']:
        src = f'{sd}/{c}_15m.json'
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)
            break

to_dl = [c for c in OLD_GREY if not os.path.exists(f'{CACHE_DIR}/{c}_15m.json')]
print(f'⬇️ تنزيل: {len(to_dl)}/{len(OLD_GREY)}')
if not to_dl:
    print('كلها موجودة!')
    exit()
print()

exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000})
t0 = time.time()

for idx, sym in enumerate(to_dl):
    fpath = f'{CACHE_DIR}/{sym}_15m.json'
    print(f'📥 {idx+1}/{len(to_dl)} {sym:<12}', end='', flush=True)
    
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
    except Exception as e:
        print(f' ❌ {e}', flush=True)

total = time.time() - t0
done = sum(1 for c in OLD_GREY if os.path.exists(f'{CACHE_DIR}/{c}_15m.json'))
print(f'\n✨ {done}/{len(OLD_GREY)} | {total/60:.0f}د')
