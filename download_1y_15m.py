#!/usr/bin/env python3
"""تحميل سنة كاملة 15m لجميع العملات الحلال مع pagination"""
import ccxt, json, os, time
from datetime import datetime, timedelta

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set(); coins = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
coins = [c for c in coins if c not in blacklist]

SAVE_DIR = '/data/trading28/data/whale_15m_1y'
os.makedirs(SAVE_DIR, exist_ok=True)
exchange = ccxt.binance({'timeout': 30000, 'enableRateLimit': True})

since_date = datetime.now() - timedelta(days=370)
print(f"📥 {len(coins)} عملة | 15m | سنة | {since_date.date()} → اليوم")
print(f"   ~{len(coins)*35000:,} شمعة متوقعة\n")

done = 0; errors = 0; total_candles = 0

for coin in coins:
    try:
        all_ohlcv = []
        since = exchange.parse8601(since_date.strftime('%Y-%m-%dT00:00:00Z'))
        while True:
            ohlcv = exchange.fetch_ohlcv(f'{coin}/USDT', '15m', since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
        
        if len(all_ohlcv) < 5000: errors += 1; continue
        
        data = {'ts': [int(o[0]) for o in all_ohlcv], 'o': [float(o[1]) for o in all_ohlcv],
                'h': [float(o[2]) for o in all_ohlcv], 'l': [float(o[3]) for o in all_ohlcv],
                'c': [float(o[4]) for o in all_ohlcv], 'v': [float(o[5]) for o in all_ohlcv]}
        
        with open(f'{SAVE_DIR}/{coin}.json', 'w') as f: json.dump(data, f)
        done += 1; total_candles += len(all_ohlcv)
        
        if done % 20 == 0:
            print(f"  {done}/{len(coins)} | {total_candles:,} شمعة", flush=True)
        time.sleep(0.05)
    except Exception as e:
        errors += 1

with open(f'{SAVE_DIR}/_manifest.json', 'w') as f:
    json.dump({'coins': done, 'errors': errors, 'total_candles': total_candles,
               'date': datetime.now().isoformat(), 'period': f'{since_date.date()} -> {datetime.now().date()}',
               'tf': '15m'}, f)

print(f"\n✅ تم: {done} عملة | {total_candles:,} شمعة | ❌ {errors}")
