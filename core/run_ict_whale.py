"""
ICT × Whale: 6 مقارنات
  MSS = Market Structure Shift
  FVG = Fair Value Gap
  LIQ = Liquidity Sweep
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

# ── المؤشرات الأساسية ──
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

# ── ICT مؤشرات ──
n = len(df)
high = df['high'].values
low = df['low'].values
close = df['close'].values

# 1. MSS (Market Structure Shift) — كسر قمة سابقة
mss_bull = np.zeros(n, dtype=bool)
for i in range(50, n):
    # نبحث عن أعلى قمة في الـ 20 شمعة السابقة
    prev_high_20 = high[i-20:i].max()
    prev_high_idx = i - 20 + np.argmax(high[i-20:i])
    # كسر القمة بشمعة قوية (إغلاق فوقها)
    if close[i] > prev_high_20 and prev_high_idx > i - 30:
        mss_bull[i] = True

# 2. FVG (Fair Value Gap) — فجوة صاعدة
fvg_bull = np.zeros(n, dtype=bool)
for i in range(3, n):
    # فجوة صاعدة: low الحالية > high قبل شمعتين
    if low[i] > high[i-2]:
        fvg_bull[i] = True

# السعر داخل منطقة FVG سابقة (قريب من الفجوة)
fvg_zone = np.zeros(n, dtype=bool)
for i in range(5, n):
    # ابحث عن آخر FVG خلال 5 شمعات
    for j in range(max(3, i-5), i):
        if fvg_bull[j]:
            fvg_top = high[j-2]
            fvg_bot = low[j]
            # السعر الحالي داخل منطقة الفجوة
            if fvg_bot <= high[i] and low[i] <= fvg_top + (fvg_top - fvg_bot) * 0.5:
                fvg_zone[i] = True
                break

# 3. Liquidity Sweep — كسر قاع سابق ثم ارتداد
liq_sweep = np.zeros(n, dtype=bool)
for i in range(20, n):
    # قاع 10 شمعات مكسور خلال آخر 3 شمعات
    prev_low_10 = low[i-10:i-3].min()
    # كسر القاع (Low) خلال آخر 3 شمعات
    if min(low[i-3:i]) < prev_low_10:
        # وارتداد: الإغلاق الحالي فوق القاع المكسور
        if close[i] > prev_low_10:
            liq_sweep[i] = True

print(f"📊 ICT signals:")
print(f"   MSS Bull: {mss_bull.sum()}")
print(f"   FVG Bull: {fvg_bull.sum()}")
print(f"   FVG Zone: {fvg_zone.sum()}")
print(f"   Liq Sweep: {liq_sweep.sum()}")

# ── إشارة أساسية ──
base_entry = (
    spike &
    (wma20 > wma50) &
    (strength > 50) &
    vol_ok &
    (df['close'] > sma50)
)

# ── السيناريوهات ──
configs = [
    ('1️⃣ الحالي (بدون ICT)', base_entry),
    ('2️⃣ + MSS (كسر قمة)', base_entry & pd.Series(mss_bull, index=df.index)),
    ('3️⃣ + FVG Zone (السعر في فجوة)', base_entry & pd.Series(fvg_zone, index=df.index)),
    ('4️⃣ + Liq Sweep (سيولة)', base_entry & pd.Series(liq_sweep, index=df.index)),
    ('5️⃣ + MSS + FVG', base_entry & pd.Series(mss_bull & fvg_zone, index=df.index)),
    ('6️⃣ + MSS + Liq + FVG (كل ICT)', base_entry & pd.Series(mss_bull & liq_sweep & fvg_zone, index=df.index)),
]

results = {}

for name, entry in configs:
    print(f"\n{'─'*70}")
    print(f"🔄 {name}")
    print(f"🚦 إشارات: {entry.sum()}")
    print(f"{'─'*70}")

    if entry.sum() == 0:
        print("   ❌ صفر إشارات")
        continue

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
print(f"🏆 ICT × Whale | FET/USDT 15m | LONG only")
print(f"{'='*95}")
print(f"{'الإعداد':<30} {'صفقات':>5} {'WR%':>6} {'المحفظة':>12} {'عائد%':>9} {'DD%':>7} {'شارب':>6}")
print(f"{'-'*95}")
for name, r in results.items():
    print(f"{name:<30} {r['trades']:>5} {r['wr']:>5.1f}% ${r['capital']:>11.2f} {r['return_pct']:>8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

print(f"\n✅ تم")
