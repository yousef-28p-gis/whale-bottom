#!/usr/bin/env python3
"""Download missing halal+halal2 coins for all 3 periods"""
import json, os, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DIRS = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'prev': '/data/trading28/data/whale_15m_prev',
    'cur': '/data/trading28/data/whale_15m_1y',
}
PERIODS = {
    '2023': (datetime(2023,9,1,tzinfo=timezone.utc), datetime(2024,9,1,tzinfo=timezone.utc)),
    'prev': (datetime(2024,9,1,tzinfo=timezone.utc), datetime(2025,9,1,tzinfo=timezone.utc)),
    'cur':  (datetime(2025,9,1,tzinfo=timezone.utc), datetime(2026,9,1,tzinfo=timezone.utc)),
}
BASE = 'https://api.binance.com/api/v3/klines'

# Load tradeable coins
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = set(d['halal'] + d['halal2'])
print(f"Tradeable coins: {len(tradeable)}")

# Find what exists
existing = {}
for period in ['2023','prev','cur']:
    existing[period] = set(f.replace('.json','') for f in os.listdir(DIRS[period]) if f.endswith('.json'))

# Build task list: only tradeable coins, only missing periods
tasks = []
for coin in sorted(tradeable):
    sym = f"{coin}USDT"
    for period in ['2023','prev','cur']:
        if coin not in existing[period]:
            tasks.append((sym, period))

print(f"Need to download: {len(tasks)} coin-periods")
print(f"  2023 missing: {len([t for t in tasks if t[1]=='2023'])}")
print(f"  prev missing: {len([t for t in tasks if t[1]=='prev'])}")
print(f"  cur missing:  {len([t for t in tasks if t[1]=='cur'])}")

if not tasks:
    print("\nNothing to download!")
else:
    def download_period(sym, period):
        coin = sym.replace('USDT','')
        out_path = os.path.join(DIRS[period], f'{coin}.json')
        start, end = PERIODS[period]
        START_TS = int(start.timestamp()*1000)
        END_TS = int(end.timestamp()*1000)
        
        all_ts, all_o, all_h, all_l, all_c = [], [], [], [], []
        since = START_TS
        while since < END_TS:
            try:
                resp = requests.get(BASE, params={
                    'symbol': sym, 'interval': '15m',
                    'startTime': since, 'limit': 1000
                }, timeout=15)
                data = resp.json()
                if not data or not isinstance(data, list): break
                for c in data:
                    if c[0] < END_TS:
                        all_ts.append(c[0]); all_o.append(float(c[1]))
                        all_h.append(float(c[2])); all_l.append(float(c[3]))
                        all_c.append(float(c[4]))
                if any(c[0] >= END_TS for c in data): break
                since = data[-1][0] + 1
                time.sleep(0.01)
            except: break
        
        if len(all_ts) >= 100:
            with open(out_path, 'w') as f:
                json.dump({'ts': all_ts, 'o': all_o, 'h': all_h, 'l': all_l, 'c': all_c}, f)
            return (coin, period, len(all_ts), 'done')
        return (coin, period, len(all_ts), 'fail' if len(all_ts) < 100 else 'partial')

    total_done, total_fail = 0, 0
    done_list, fail_list = [], []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(download_period, s, p): (s,p) for s,p in tasks}
        for i, f in enumerate(as_completed(futures)):
            coin, period, n, status = f.result()
            if status == 'done':
                total_done += n
                done_list.append(f"{coin}/{period}")
            else:
                total_fail += 1
                fail_list.append(f"{coin}/{period} ({n} candles)")
            if (i+1) % 20 == 0:
                print(f'  {i+1}/{len(tasks)}  done={total_done}  fail={total_fail}', flush=True)

    print(f'\nDone: {total_done} candles, Failed: {total_fail}')
    if fail_list:
        print(f"\nFailed ({len(fail_list)}):")
        for f in fail_list:
            print(f"  {f}")

    # Final check
    print("\n=== FINAL ===")
    coins = {}
    for period in ['2023','prev','cur']:
        s = set(f.replace('.json','') for f in os.listdir(DIRS[period]) if f.endswith('.json'))
        coins[period] = s
        have = tradeable & s
        print(f"{period}: {len(have)}/{len(tradeable)} tradeable")

    common = coins['2023'] & coins['prev'] & coins['cur'] & tradeable
    print(f"\nCommon (all 3 + tradeable): {len(common)} / 212")
    if len(common) < 212:
        missing = tradeable - common
        print(f"Still missing: {len(missing)}")
        for c in sorted(missing)[:30]:
            print(f"  {c}")
        if len(missing) > 30:
            print(f"  ... +{len(missing)-30} more")
    else:
        print("✅ ALL 212 COMPLETE!")
