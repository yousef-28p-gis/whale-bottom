#!/usr/bin/env python3
"""Deeper whale analysis — what threshold matters?"""
import json
import numpy as np
from collections import Counter

with open('/data/trading28/backtests/pattern_data/pre_pump_data.json') as f:
    pre_pump = json.load(f)

all_entries = []
for date_str, coins in pre_pump.items():
    for coin, ind in coins.items():
        ind['date'] = date_str
        ind['coin'] = coin
        all_entries.append(ind)

print(f"Total entries: {len(all_entries)}")

# Whale bar distribution
whale_counts = [e['whale_bars_count'] for e in all_entries]
whale_ratios = [e['whale_ratio'] for e in all_entries]

print(f"\n## Whale Bar Distribution")
print(f"   Whale bars count: mean={np.mean(whale_counts):.1f}, median={np.median(whale_counts):.0f}")
print(f"   Whale ratio: mean={np.mean(whale_ratios):.3f}, median={np.median(whale_ratios):.3f}")

# Distribution by count
for threshold in [0, 1, 2, 3, 5, 7, 10]:
    n = sum(1 for w in whale_counts if w >= threshold)
    print(f"   >= {threshold} whale bars: {n} ({n/len(whale_counts)*100:.0f}%)")

# What if we set higher threshold — say >= 3 whale bars?
print(f"\n{'='*60}")
print(f"🔬 COMPARING: >=3 whale bars vs <3 whale bars")
print(f"{'='*60}")

high_whale = [e for e in all_entries if e['whale_bars_count'] >= 3]
low_whale = [e for e in all_entries if e['whale_bars_count'] < 3]

print(f"\n>=3 whales: {len(high_whale)} pumps")
print(f"<3 whales:  {len(low_whale)} pumps")

if low_whale:
    print(f"\n## <3 WHALES — What else do they share?")
    
    # RSI
    rsi = [e['last_rsi'] for e in low_whale if e.get('last_rsi') is not None]
    print(f"\n### RSI")
    print(f"   Mean: {np.mean(rsi):.1f}, Median: {np.median(rsi):.1f}")
    print(f"   <30: {sum(1 for r in rsi if r<30)} ({sum(1 for r in rsi if r<30)/len(rsi)*100:.0f}%)")
    print(f"   <40: {sum(1 for r in rsi if r<40)} ({sum(1 for r in rsi if r<40)/len(rsi)*100:.0f}%)")
    
    # Prev day
    prev = [e['prev_day_pct'] for e in low_whale if e.get('prev_day_pct') is not None]
    print(f"\n### Previous Day")
    red = sum(1 for p in prev if p < 0)
    print(f"   RED: {red} ({red/len(prev)*100:.0f}%)")
    
    # Price position
    pos = [e['price_position'] for e in low_whale if e.get('price_position') is not None]
    print(f"\n### Price Position")
    bottom = sum(1 for p in pos if p < 0.3)
    print(f"   Bottom: {bottom} ({bottom/len(pos)*100:.0f}%)")
    
    # Days red
    dr = [e['days_red_before'] for e in low_whale if e.get('days_red_before') is not None]
    print(f"\n### Red Days Before")
    print(f"   Mean: {np.mean(dr):.1f}")
    print(f"   2+: {sum(1 for d in dr if d>=2)} ({sum(1 for d in dr if d>=2)/len(dr)*100:.0f}%)")

# Also check: what does >=5 whales look like?
print(f"\n{'='*60}")
print(f"🔬 >=5 WHALE BARS — The strongest signal")
print(f"{'='*60}")
very_whale = [e for e in all_entries if e['whale_bars_count'] >= 5]
print(f"Count: {len(very_whale)} ({len(very_whale)/len(all_entries)*100:.0f}%)")
avg_pump_very = np.mean([e['pump_pct'] for e in very_whale])
avg_pump_all = np.mean([e['pump_pct'] for e in all_entries])
print(f"Avg pump: +{avg_pump_very:.1f}% vs all +{avg_pump_all:.1f}%")

# RSI buckets and whale correlation
print(f"\n{'='*60}")
print(f"🔬 RSI vs Whale Activity")
print(f"{'='*60}")
for rsi_range, (lo, hi) in [('<30', (0,30)), ('30-40', (30,40)), ('40-50', (40,50)), ('50-60', (50,60)), ('60-70', (60,70)), ('>70', (70,100))]:
    in_range = [e for e in all_entries if e.get('last_rsi') and lo <= e['last_rsi'] < hi]
    if in_range:
        avg_w = np.mean([e['whale_bars_count'] for e in in_range])
        avg_p = np.mean([e['pump_pct'] for e in in_range])
        print(f"   RSI {rsi_range:8s}: {len(in_range)} pumps, avg whale={avg_w:.1f}, avg pump=+{avg_p:.1f}%")

# Final: what combination gives highest pump?
print(f"\n{'='*60}")
print(f"🏆 BEST COMBINATIONS (highest avg pump)")
print(f"{'='*60}")

combos = [
    ("Whale>=5 + RSI<40", lambda e: e['whale_bars_count']>=5 and e.get('last_rsi') and e['last_rsi']<40),
    ("Whale>=5 + PrevRed", lambda e: e['whale_bars_count']>=5 and e.get('prev_day_pct') is not None and e['prev_day_pct']<0),
    ("Whale>=3 + RSI<30", lambda e: e['whale_bars_count']>=3 and e.get('last_rsi') and e['last_rsi']<30),
    ("Whale>=3 + Bottom", lambda e: e['whale_bars_count']>=3 and e.get('price_position') and e['price_position']<0.3),
    ("Whale>=5 + RSI<40 + Bottom", lambda e: e['whale_bars_count']>=5 and e.get('last_rsi') and e['last_rsi']<40 and e.get('price_position') and e['price_position']<0.3),
    ("Whale>=3 + 2+ RedDays", lambda e: e['whale_bars_count']>=3 and e.get('days_red_before') and e['days_red_before']>=2),
    ("Whale>=3 + RSI<40 + PrevRed", lambda e: e['whale_bars_count']>=3 and e.get('last_rsi') and e['last_rsi']<40 and e.get('prev_day_pct') is not None and e['prev_day_pct']<0),
]

for name, fn in combos:
    matches = [e for e in all_entries if fn(e)]
    if matches:
        avg_p = np.mean([e['pump_pct'] for e in matches])
        print(f"   {name}: {len(matches)} pumps, avg +{avg_p:.1f}%")
