import ccxt
import json
from datetime import datetime, timezone

exchange = ccxt.binance()

since = exchange.parse8601('2022-11-21T00:00:00Z')

all_candles = []
while True:
    candles = exchange.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not candles:
        break
    all_candles.extend(candles)
    since = candles[-1][0] + 1
    if candles[-1][0] > exchange.parse8601('2022-12-01T00:00:00Z'):
        break
    if len(candles) >= 3000:
        break

print(f"Total 5m candles: {len(all_candles)}")
print(f"First: {datetime.fromtimestamp(all_candles[0][0]/1000, tz=timezone.utc)}")
print(f"Last: {datetime.fromtimestamp(all_candles[-1][0]/1000, tz=timezone.utc)}")
print(f"Min low: {min(c[3] for c in all_candles)}")
print(f"Max high: {max(c[2] for c in all_candles)}")

with open('/data/trading28/data_btc_5m_nov2022.json', 'w') as f:
    json.dump(all_candles, f)

print("Saved to data_btc_5m_nov2022.json")
