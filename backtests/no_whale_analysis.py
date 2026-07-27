#!/usr/bin/env python3
"""Analyze WITHOUT whales - what common factors predict pumps?"""
import json
import numpy as np
from collections import Counter

with open('/data/trading28/backtests/pattern_data/pre_pump_data.json') as f:
    pre_pump = json.load(f)

# Split
with_whale = []
without_whale = []

for date_str, coins in pre_pump.items():
    for coin, ind in coins.items():
        entry = {'date': date_str, 'coin': coin, **ind}
        if ind.get('whale_bars_count', 0) > 0:
            with_whale.append(entry)
        else:
            without_whale.append(entry)

N = len(with_whale) + len(without_whale)
print(f"{'='*60}")
print(f"📊 WITH WHALES:    {len(with_whale)} ({len(with_whale)/N*100:.0f}%)")
print(f"📊 WITHOUT WHALES: {len(without_whale)} ({len(without_whale)/N*100:.0f}%)")
print(f"{'='*60}")

ww = without_whale

# Pump stats
pumps = [e['pump_pct'] for e in ww if e.get('pump_pct')]
print(f"\n## Pump Size (بدون حيتان)")
print(f"   Avg: +{np.mean(pumps):.1f}% | Median: +{np.median(pumps):.1f}% | Max: +{np.max(pumps):.1f}%")

# RSI
rsi = [e['last_rsi'] for e in ww if e.get('last_rsi') is not None]
print(f"\n## 1️⃣ RSI")
print(f"   Mean: {np.mean(rsi):.1f} | Median: {np.median(rsi):.1f}")
oversold = sum(1 for r in rsi if r < 30)
near_os = sum(1 for r in rsi if 30 <= r < 40)
mid = sum(1 for r in rsi if 40 <= r <= 60)
over = sum(1 for r in rsi if r > 70)
print(f"   <30 (oversold):   {oversold} ({oversold/len(rsi)*100:.0f}%) ⭐")
print(f"   30-40 (near OS):  {near_os} ({near_os/len(rsi)*100:.0f}%)")
print(f"   40-60 (neutral):  {mid} ({mid/len(rsi)*100:.0f}%)")
print(f"   >70 (overbought): {over} ({over/len(rsi)*100:.0f}%)")

# Volume
vol_ratio = [e['vol_ratio_4h'] for e in ww if e.get('vol_ratio_4h') is not None]
print(f"\n## 2️⃣ Volume Ratio 4h/24h")
print(f"   Mean: {np.mean(vol_ratio):.2f}")
high = sum(1 for v in vol_ratio if v > 1.5)
low = sum(1 for v in vol_ratio if v < 0.5)
print(f"   >1.5x (high): {high} ({high/len(vol_ratio)*100:.0f}%)")
print(f"   <0.5x (low):  {low} ({low/len(vol_ratio)*100:.0f}%)")

# Price position
pos = [e['price_position'] for e in ww if e.get('price_position') is not None]
print(f"\n## 3️⃣ Price Position in Range")
print(f"   Mean: {np.mean(pos):.2f}")
bottom = sum(1 for p in pos if p < 0.3)
mid_r = sum(1 for p in pos if 0.3 <= p <= 0.7)
top = sum(1 for p in pos if p > 0.7)
print(f"   Bottom (<0.3):  {bottom} ({bottom/len(pos)*100:.0f}%) ⭐")
print(f"   Mid (0.3-0.7):  {mid_r} ({mid_r/len(pos)*100:.0f}%)")
print(f"   Top (>0.7):     {top} ({top/len(pos)*100:.0f}%)")

# 8h change
p8h = [e['price_8h_change'] for e in ww if e.get('price_8h_change') is not None]
print(f"\n## 4️⃣ Price 8h Change")
print(f"   Mean: {np.mean(p8h):.2f}%")
fall = sum(1 for p in p8h if p < -1)
flat = sum(1 for p in p8h if -1 <= p <= 1)
rise = sum(1 for p in p8h if p > 1)
print(f"   Falling: {fall} ({fall/len(p8h)*100:.0f}%)")
print(f"   Flat:    {flat} ({flat/len(p8h)*100:.0f}%)")
print(f"   Rising:  {rise} ({rise/len(p8h)*100:.0f}%)")

# Previous day
prev_pct = [e['prev_day_pct'] for e in ww if e.get('prev_day_pct') is not None]
print(f"\n## 5️⃣ Previous Day")
print(f"   Mean: {np.mean(prev_pct):.2f}%")
prev_red = sum(1 for p in prev_pct if p < 0)
prev_green = sum(1 for p in prev_pct if p > 0)
print(f"   RED:   {prev_red} ({prev_red/len(prev_pct)*100:.0f}%) ⭐")
print(f"   GREEN: {prev_green} ({prev_green/len(prev_pct)*100:.0f}%)")

# Days red
days_red = [e['days_red_before'] for e in ww if e.get('days_red_before') is not None]
print(f"\n   Mean red days before: {np.mean(days_red):.1f}")
r2 = sum(1 for d in days_red if d >= 2)
r3 = sum(1 for d in days_red if d >= 3)
print(f"   2+ red days: {r2} ({r2/len(days_red)*100:.0f}%) ⭐")
print(f"   3+ red days: {r3} ({r3/len(days_red)*100:.0f}%)")

# Candles
green_c = [e['green_of_last_6h'] for e in ww if e.get('green_of_last_6h') is not None]
red_c = [e['red_of_last_6h'] for e in ww if e.get('red_of_last_6h') is not None]
print(f"\n## 6️⃣ Last 6h Candles")
print(f"   Green mean: {np.mean(green_c):.1f}/6, Red mean: {np.mean(red_c):.1f}/6")
mostly_red = sum(1 for i in range(len(green_c)) if (red_c[i] - green_c[i]) >= 2)
print(f"   Mostly RED: {mostly_red} ({mostly_red/len(green_c)*100:.0f}%)")

# Top coins
print(f"\n## 7️⃣ Most Frequent (بدون حيتان)")
coin_counts = Counter(e['coin'] for e in ww)
for coin, count in coin_counts.most_common(10):
    avg = np.mean([e['pump_pct'] for e in ww if e['coin'] == coin])
    print(f"   {coin}: {count}x, avg +{avg:.1f}%")

# ── COMPARISON ──
print(f"\n{'='*60}")
print(f"🔬 WITH WHALES vs WITHOUT — What differs?")
print(f"{'='*60}")

comps = [
    ('RSI', 'last_rsi'),
    ('Vol Ratio', 'vol_ratio_4h'),
    ('Vol Trend', 'vol_trend'),
    ('Price Pos', 'price_position'),
    ('8h Chg', 'price_8h_change'),
    ('Prev Day%', 'prev_day_pct'),
    ('Days Red', 'days_red_before'),
    ('Green/6h', 'green_of_last_6h'),
    ('Red/6h', 'red_of_last_6h'),
]
for name, col in comps:
    w_vals = [e[col] for e in with_whale if e.get(col) is not None]
    wo_vals = [e[col] for e in without_whale if e.get(col) is not None]
    if w_vals and wo_vals:
        diff = np.mean(w_vals) - np.mean(wo_vals)
        print(f"   {name:15s}: With🐋 {np.mean(w_vals):.2f}  vs  Without {np.mean(wo_vals):.2f}  (Δ{diff:+.2f})")

# ── KEY INSIGHT ──
print(f"\n{'='*60}")
print(f"🎯 KEY INSIGHT: What predicts pumps WITHOUT whales?")
print(f"{'='*60}")
print(f"   1. RSI near oversold or low (<40): {oversold+near_os}/{len(rsi)} = {(oversold+near_os)/len(rsi)*100:.0f}%")
print(f"   2. Previous day RED: {prev_red}/{len(prev_pct)} = {prev_red/len(prev_pct)*100:.0f}%")
print(f"   3. Price at bottom of range: {bottom}/{len(pos)} = {bottom/len(pos)*100:.0f}%")
print(f"   4. 2+ red days before: {r2}/{len(days_red)} = {r2/len(days_red)*100:.0f}%")
