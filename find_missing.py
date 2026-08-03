import os

dirs = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR': '/data/trading28/data/whale_15m_1y',
}

coins = {}
for name, path in dirs.items():
    coins[name] = set(f.replace('.json','') for f in os.listdir(path) if f.endswith('.json'))

common = coins['2023'] & coins['PREV'] & coins['CUR']
print(f"Common: {len(common)}")

# Missing from 2023 but present in PREV and CUR (these are our best candidates)
candidates = (coins['PREV'] & coins['CUR']) - coins['2023']
print(f"\nIn PREV+CUR but missing from 2023: {len(candidates)} coins")
# Sort and show
for c in sorted(candidates):
    print(f"  {c}")
