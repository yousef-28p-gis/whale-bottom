import os

dirs = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR': '/data/trading28/data/whale_15m_1y',
}

coins = {}
for name, path in dirs.items():
    coins[name] = set(f.replace('.json','') for f in os.listdir(path) if f.endswith('.json'))

common_3 = coins['2023'] & coins['PREV'] & coins['CUR']
print(f"Current: {len(common_3)} common out of target 212")
print(f"Need: {212 - len(common_3)} more\n")

# Which are in CUR but missing from 2023
need_2023 = coins['CUR'] - coins['2023']
print(f"Need to download for 2023: {len(need_2023)} coins")
# Which are in CUR but missing from PREV
need_prev = coins['CUR'] - coins['PREV']
print(f"Need to download for PREV: {len(need_prev)} coins")

# The combined set: need at least one of these
need_both = need_2023 | need_prev
print(f"\nTotal coins to try downloading: {len(need_both)}")

# Show first 30 of each
print("\n--- Sample missing from 2023 ---")
for c in sorted(need_2023)[:30]:
    print(f"  {c}")
if len(need_2023) > 30:
    print(f"  ... +{len(need_2023)-30} more")

print("\n--- Sample missing from PREV ---")
for c in sorted(need_prev)[:30]:
    print(f"  {c}")
if len(need_prev) > 30:
    print(f"  ... +{len(need_prev)-30} more")
