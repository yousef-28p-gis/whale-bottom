"""
تشغيل الاستراتيجية على بيانات FET كاملة مع تدقيق آلي.
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

# ============================================================
# تحميل البيانات
# ============================================================
DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📦 {len(df)} شمعة | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# ============================================================
# المؤشرات
# ============================================================
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

# إشارة الدخول
entry_signal = (
    spike &
    (wma20 > wma50) &
    (strength > 50) &
    vol_ok &
    (df['close'] > sma50)
)

print(f"🚦 إشارات الدخول: {entry_signal.sum()}")

# ============================================================
# تشغيل الاستراتيجيتين
# ============================================================
results = {}

for tp_mode, label in [('ema21', 'EMA21 + Swing SL'), ('3atr', '3ATR + Swing SL')]:
    print(f"\n{'─'*60}")
    print(f"🔄 {label}")
    print(f"{'─'*60}")
    
    r = run_backtest(
        df=df,
        entry_signal=entry_signal,
        tp_series=ema,
        atr_series=atr_val,
        sell_series=sell,
        swing_mask=sw_mask,
        sma50_series=sma50,
        tp_mode=tp_mode,
        max_hours=48,
        monthly_limit=0.07,
        fee=0.001,
    )
    
    if 'error' in r:
        print(f"❌ {r['error']}")
        continue
    
    # تدقيق
    ok = verify_trades(df, r)
    if not ok:
        print("⛔ توقف — فيه أخطاء!")
        continue
    
    results[label] = r
    
    # عرض النتائج
    tdf = r['tdf']
    print(f"\n📊 {label}:")
    print(f"   صفقات: {r['trades']} | WR: {r['wr']:.1f}%")
    print(f"   متوسط ربح: +{r['avg_win']:.2f}% | متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"   أقصى ربح: +{r['max_win']:.2f}% | أقصى خسارة: {r['max_loss']:.2f}%")
    print(f"   محفظة: $1000 → ${r['capital']:.2f} (+{r['return_pct']:.1f}%)")
    print(f"   DD: {r['dd']:.1f}% | Sharpe: {r['sharpe']:.2f} | مدة: {r['avg_dur']:.0f}د")
    
    # مخارج
    print(f"   مخارج:", {k: v for k, v in tdf['exit_reason'].value_counts().items()})
    
    # سنوي
    print(f"   سنوي:")
    for yr, grp in tdf.groupby('year'):
        ycap = 1.0
        for _, t in grp.iterrows():
            ycap *= (1 + t['pnl_pct'] / 100)
        yret = (ycap - 1) * 100
        ywr = (grp['pnl_pct'] > 0).sum() / len(grp) * 100
        print(f"     {yr}: {len(grp)}T | WR={ywr:.0f}% | {'+' if yret>=0 else ''}{yret:.1f}% {'✅' if yret>=0 else '❌'}")

# ============================================================
# مقارنة
# ============================================================
if len(results) == 2:
    print(f"\n{'='*60}")
    print(f"🏆 المقارنة النهائية")
    print(f"{'='*60}")
    print(f"{'الاستراتيجية':<25} {'صفقات':>6} {'WR%':>6} {'المحفظة':>12} {'عائد%':>9} {'DD%':>7} {'شارب':>6}")
    print(f"{'-'*60}")
    for label, r in results.items():
        print(f"{label:<25} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>11.2f} {r['return_pct']:>8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")
