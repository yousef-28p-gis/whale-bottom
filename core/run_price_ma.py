"""
فلتر اتجاه السعر: ٤ متوسطات مختلفة
"""
import sys
sys.path.insert(0, '/data/trading28')

import pandas as pd
import numpy as np
from core.indicators import (
    whale_indicator, whale_ma, whale_strength, whale_spike,
    volume_filter, sma50_daily, ema21, atr, sell_signal, swing_lows
)
from core.backtest_engine import run_backtest

DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📦 {len(df)} شمعة | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# المؤشرات
whale = whale_indicator(df, 200)
wma20 = whale_ma(whale, 20)
wma50 = whale_ma(whale, 50)
strength = whale_strength(whale, 50)
spike = whale_spike(whale)
vol_ok = volume_filter(df)
sma50 = sma50_daily(df)
ema = ema21(df)
atr_val = atr(df, 14)
sell = sell_signal(df)
sw_mask = swing_lows(df, 5)

# متوسطات السعر
price_ema20 = df['close'].ewm(span=20, adjust=False).mean()
price_ema50 = df['close'].ewm(span=50, adjust=False).mean()
price_sma50 = df['close'].rolling(50).mean()
price_sma200 = df['close'].rolling(200).mean()

# إشارة أساسية (بدون فلتر السعر الإضافي)
base_entry = (
    spike &
    (wma20 > wma50) &
    (strength > 50) &
    vol_ok &
    (df['close'] > sma50)
)

print(f"🚦 الإشارات الأساسية: {base_entry.sum()}")

configs = [
    ('الحالي (SMA50 يومي فقط)', base_entry),
    ('➕ EMA20 > EMA50 (سعر)', base_entry & (price_ema20 > price_ema50)),
    ('➕ SMA50 > SMA200 (تقاطع ذهبي)', base_entry & (price_sma50 > price_sma200)),
    ('➕ سعر > EMA50', base_entry & (df['close'] > price_ema50)),
    ('➕ سعر > SMA200', base_entry & (df['close'] > price_sma200)),
]

results = {}

for name, entry in configs:
    print(f"\n{'─'*70}")
    print(f"🔄 {name}")
    print(f"🚦 إشارات: {entry.sum()}")
    print(f"{'─'*70}")

    r = run_backtest(df=df, entry_signal=entry, tp_series=ema, atr_series=atr_val,
                     sell_series=sell, swing_mask=sw_mask, sma50_series=sma50,
                     tp_mode='ema21', max_hours=48, monthly_limit=0.07, fee=0.001)

    if 'error' in r:
        print(f"❌ {r['error']}")
        continue

    results[name] = r
    tdf = r['tdf']

    print(f"   صفقات: {r['trades']} | WR: {r['wr']:.1f}%")
    print(f"   متوسط ربح: +{r['avg_win']:.2f}% | متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"   محفظة: $1000 → ${r['capital']:.2f} (+{r['return_pct']:.1f}%)")
    print(f"   DD: {r['dd']:.1f}% | Sharpe: {r['sharpe']:.2f} | مدة: {r['avg_dur']:.0f}د")

    # سنوي
    print(f"   سنوي:", end="")
    for yr, grp in tdf.groupby('year'):
        ycap = 1.0
        for _, t in grp.iterrows():
            ycap *= (1 + t['pnl_pct'] / 100)
        yret = (ycap - 1) * 100
        ywr = (grp['pnl_pct'] > 0).sum() / len(grp) * 100
        print(f" {yr}:{len(grp)}T/{ywr:.0f}%/{yret:+.0f}%", end="")
    print()

print(f"\n{'='*95}")
print(f"🏆 فلتر اتجاه السعر | FET/USDT 15m | LONG only")
print(f"{'='*95}")
print(f"{'الإعداد':<38} {'صفقات':>5} {'WR%':>6} {'المحفظة':>12} {'عائد%':>9} {'DD%':>7} {'شارب':>6}")
print(f"{'-'*95}")
for name, r in results.items():
    print(f"{name:<38} {r['trades']:>5} {r['wr']:>5.1f}% ${r['capital']:>11.2f} {r['return_pct']:>8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

print(f"\n✅ تم")
