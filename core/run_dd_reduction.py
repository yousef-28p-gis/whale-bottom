"""
مقارنة: الإعدادات الحالية vs تعديلات خفض DD الثلاثة
  ١. قوة الحوت > 70% (بدل 50%)
  ٢. فلتر ATR: ATR(14) < 1.5 × ATR(50) — منع الدخول في التقلب العالي
  ٣. حد خسارة شهري 10%
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
from core.verify import verify_trades

# ── تحميل البيانات ──
DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📦 {len(df)} شمعة | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# ── المؤشرات ──
print("⏳ حساب المؤشرات...")

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

# فلتر ATR — منع التقلب العالي
atr_ma50 = atr_val.rolling(50).mean()
atr_ok = atr_val < atr_ma50 * 1.5  # ATR أقل من 1.5× متوسطه

# ── اختبار ٤ إعدادات ──
configs = [
    {
        'name': '1️⃣ الإعداد الحالي (قوة>50%، بدون ATR، حد 7%)',
        'strength_min': 50,
        'use_atr': False,
        'monthly_limit': 0.07,
    },
    {
        'name': '2️⃣ قوة>70% فقط',
        'strength_min': 70,
        'use_atr': False,
        'monthly_limit': 0.07,
    },
    {
        'name': '3️⃣ ATR فلتر فقط',
        'strength_min': 50,
        'use_atr': True,
        'monthly_limit': 0.07,
    },
    {
        'name': '🔥 الثلاثة معاً: قوة>70% + ATR + حد 10%',
        'strength_min': 70,
        'use_atr': True,
        'monthly_limit': 0.10,
    },
]

results = {}

for cfg in configs:
    print(f"\n{'─'*70}")
    print(f"🔄 {cfg['name']}")
    print(f"{'─'*70}")

    # بناء إشارة الدخول
    entry_signal = (
        spike &
        (wma20 > wma50) &
        (strength > cfg['strength_min']) &
        vol_ok &
        (df['close'] > sma50)
    )
    
    if cfg['use_atr']:
        entry_signal = entry_signal & atr_ok

    print(f"🚦 إشارات الدخول: {entry_signal.sum()}")

    r = run_backtest(
        df=df,
        entry_signal=entry_signal,
        tp_series=ema,
        atr_series=atr_val,
        sell_series=sell,
        swing_mask=sw_mask,
        sma50_series=sma50,
        tp_mode='ema21',
        max_hours=48,
        monthly_limit=cfg['monthly_limit'],
        fee=0.001,
    )

    if 'error' in r:
        print(f"❌ {r['error']}")
        continue

    ok = verify_trades(df, r)
    if not ok:
        print("⛔ توقف — فيه أخطاء!")
        continue

    results[cfg['name']] = r

    tdf = r['tdf']
    print(f"\n📊 {cfg['name']}:")
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

# ── جدول مقارنة ──
print(f"\n{'='*90}")
print(f"🏆 مقارنة شاملة | FET/USDT 15m | EMA21 + Swing SL")
print(f"{'='*90}")
print(f"{'الإعداد':<35} {'صفقات':>5} {'WR%':>6} {'المحفظة':>12} {'عائد%':>9} {'DD%':>7} {'شارب':>6}")
print(f"{'-'*90}")
for name, r in results.items():
    print(f"{name:<35} {r['trades']:>5} {r['wr']:>5.1f}% ${r['capital']:>11.2f} {r['return_pct']:>8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

print(f"\n✅ تم")
