"""اختبار المتوسطات على مؤشر الحوت"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd
import numpy as np
from core.indicators import *
from core.backtest_engine import run_backtest

DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

# مؤشرات أساسية
whale = whale_indicator(df, 200)
strength = whale_strength(whale, 50)
spike = whale_spike(whale)
vol_ok = volume_filter(df)
sma50 = sma50_daily(df)
ema = ema21(df)
atr_val = atr(df, 14)
sell = sell_signal(df)
sw_mask = swing_lows(df, 5)

# المتوسطات المختلفة
periods = [10, 20, 50, 100, 200, 500, 1000]
mas = {}
for p in periods:
    mas[p] = whale_ma(whale, p)

print("=" * 70)
print("🔬 اختبار المتوسطات على مؤشر الحوت")
print("=" * 70)

results = []

# ١. متوسطين (كل التركيبات)
print(f"\n📊 متوسطين — كل التركيبات:")
print(f"{'Fast':>6} {'Slow':>6} {'صفقات':>6} {'WR%':>6} {'محفظة':>10} {'DD%':>7} {'شارب':>6}")
print("-" * 55)

for fast_p in periods[:-1]:
    for slow_p in periods:
        if slow_p <= fast_p:
            continue
        trend = mas[fast_p] > mas[slow_p]
        sig = spike & trend & (strength > 50) & vol_ok & (df['close'] > sma50)
        r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
        if 'error' not in r:
            results.append({
                'type': '2MA', 'fast': fast_p, 'slow': slow_p,
                'trades': r['trades'], 'wr': r['wr'], 'capital': r['capital'],
                'dd': r['dd'], 'sharpe': r['sharpe'],
            })
            marker = '⭐' if r['capital'] > 2500 else '  '
            print(f"{marker} {fast_p:>5} {slow_p:>5} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>9.0f} {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

# ٢. ٣ متوسطات
print(f"\n📊 ٣ متوسطات:")
print(f"{'MA1':>5} {'MA2':>5} {'MA3':>5} {'صفقات':>6} {'WR%':>6} {'محفظة':>10} {'DD%':>7} {'شارب':>6}")
print("-" * 60)

triplets = [
    (10, 20, 50), (10, 50, 100), (10, 100, 200),
    (20, 50, 100), (20, 50, 200), (20, 100, 200),
    (50, 100, 200), (50, 200, 500), (50, 100, 500),
    (100, 200, 500), (100, 500, 1000),
]

for a, b, c in triplets:
    trend = (mas[a] > mas[b]) & (mas[b] > mas[c])
    sig = spike & trend & (strength > 50) & vol_ok & (df['close'] > sma50)
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
    if 'error' not in r:
        results.append({
            'type': '3MA', 'fast': a, 'mid': b, 'slow': c,
            'trades': r['trades'], 'wr': r['wr'], 'capital': r['capital'],
            'dd': r['dd'], 'sharpe': r['sharpe'],
        })
        marker = '⭐' if r['capital'] > 2500 else '  '
        print(f"{marker} {a:>5} {b:>5} {c:>5} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>9.0f} {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

# ٣. متوسط واحد (فوق/تحت MA)
print(f"\n📊 متوسط واحد (السعر المتحرك للحوت):")
print(f"{'MA':>6} {'صفقات':>6} {'WR%':>6} {'محفظة':>10} {'DD%':>7} {'شارب':>6}")
print("-" * 45)

for p in [20, 50, 100, 200, 500, 1000]:
    # الحوت الحالي > متوسطه (نشاط الحيتان يتسارع)
    trend = whale > mas[p]
    sig = spike & trend & (strength > 50) & vol_ok & (df['close'] > sma50)
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
    if 'error' not in r:
        results.append({
            'type': '1MA', 'period': p,
            'trades': r['trades'], 'wr': r['wr'], 'capital': r['capital'],
            'dd': r['dd'], 'sharpe': r['sharpe'],
        })
        marker = '⭐' if r['capital'] > 2500 else '  '
        print(f"{marker} {p:>5} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>9.0f} {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

# ٤. بدون فلتر ترند ( baseline)
print(f"\n📊 بدون فلتر ترند (للمقارنة):")
sig = spike & (strength > 50) & vol_ok & (df['close'] > sma50)
r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
if 'error' not in r:
    print(f"   بدون: {r['trades']:>6}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")

# ترتيب أفضل ٥
print(f"\n{'='*70}")
print(f"🏆 أفضل ٥ تركيبات:")
print(f"{'='*70}")

sorted_results = sorted(results, key=lambda x: x['capital'], reverse=True)
for i, r in enumerate(sorted_results[:5]):
    if r['type'] == '2MA':
        desc = f"wMA{r['fast']} > wMA{r['slow']}"
    elif r['type'] == '3MA':
        desc = f"wMA{r['fast']} > wMA{r['mid']} > wMA{r['slow']}"
    else:
        desc = f"حوت > wMA{r['period']}"
    
    print(f"   {i+1}. {desc:<30s} {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")
