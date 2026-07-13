"""
فلتر منع الدخول أثناء ومضة التصريف
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

# المؤشرات الأساسية
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

# ومضة تصريف
lookback = 200
highest_n = df['high'].rolling(lookback).max()
at_high = df['high'] >= highest_n
high_change = abs(df['high'] - df['high'].shift(1)) / df['high'] * 100
smooth_change_h = high_change.ewm(span=3, adjust=False).mean()
highest_change_h = smooth_change_h.rolling(lookback).max()
strength_h = np.where(at_high, (smooth_change_h + highest_change_h * 2) / 3, 0)
dump_raw = pd.Series(strength_h, index=df.index).ewm(span=3, adjust=False).mean().fillna(0)

dump_spike = (dump_raw > dump_raw.shift(1)) & (dump_raw.shift(1) <= 0.02)
dump_strength = pd.Series(
    np.where(dump_raw.rolling(50).max() > 0, dump_raw / dump_raw.rolling(50).max() * 100, 0),
    index=df.index
)

dump_active = dump_spike & (dump_strength > 50)

# إشارة دخول أساسية
base_entry = (
    spike &
    (wma20 > wma50) &
    (strength > 50) &
    vol_ok &
    (df['close'] > sma50)
)

print(f"🚦 الإشارات الأساسية: {base_entry.sum()}")
print(f"🛑 أوقات التصريف النشط: {dump_active.sum()}")
print(f"📊 إشارات وقت التصريف (ممنوعة): {(base_entry & dump_active).sum()}")

configs = [
    ('الحالي (بدون فلتر التصريف)', base_entry),
    ('مع فلتر التصريف (لا دخول أثناء التوزيع)', base_entry & ~dump_active),
]

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

print(f"\n✅ تم")
