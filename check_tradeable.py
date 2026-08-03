import json, os

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = set(d['halal'] + d['halal2'])
print(f"Target: {len(tradeable)} tradeable coins\n")

dirs = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR': '/data/trading28/data/whale_15m_1y',
}

coins = {}
for name, path in dirs.items():
    coins[name] = set(f.replace('.json','') for f in os.listdir(path) if f.endswith('.json'))
    have = tradeable & coins[name]
    missing = tradeable - coins[name]
    print(f"{name}: {len(have)}/{len(tradeable)} have, {len(missing)} missing")

common = coins['2023'] & coins['PREV'] & coins['CUR'] & tradeable
print(f"\nCommon (all 3 + tradeable): {len(common)}")
print(f"Still need: {212 - len(common)}")

# Missing from 2023 specifically
missing_2023 = tradeable - coins['2023']
print(f"\nMissing from 2023 ({len(missing_2023)}):")
for c in sorted(missing_2023):
    print(f"  {c}")

# Missing from PREV
missing_prev = tradeable - coins['PREV']
print(f"\nMissing from PREV ({len(missing_prev)}):")
for c in sorted(missing_prev):
    print(f"  {c}")

# Missing from CUR
missing_cur = tradeable - coins['CUR']
print(f"\nMissing from CUR ({len(missing_cur)}):")
for c in sorted(missing_cur):
    print(f"  {c}")
