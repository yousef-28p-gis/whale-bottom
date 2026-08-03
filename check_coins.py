import os

dirs = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR': '/data/trading28/data/whale_15m_1y',
}

coins = {}
for name, path in dirs.items():
    coins[name] = set(f.replace('.json','') for f in os.listdir(path) if f.endswith('.json'))

for name, s in coins.items():
    print(f"{name}: {len(s)} coins")

common_3 = coins['2023'] & coins['PREV'] & coins['CUR']
print(f"\nAll 3 years: {len(common_3)} coins")

# Missing from 2023 but available in PREV+CUR
missing_2023 = (coins['PREV'] & coins['CUR']) - coins['2023']
print(f"\nMissing from 2023 (in PREV+CUR but not 2023): {len(missing_2023)}")
for c in sorted(missing_2023):
    print(f"  {c}")

# Missing from PREV
missing_prev = (coins['2023'] & coins['CUR']) - coins['PREV']
print(f"\nMissing from PREV: {len(missing_prev)}")
for c in sorted(missing_prev):
    print(f"  {c}")

# Missing from CUR
missing_cur = (coins['2023'] & coins['PREV']) - coins['CUR']
print(f"\nMissing from CUR: {len(missing_cur)}")
for c in sorted(missing_cur):
    print(f"  {c}")
