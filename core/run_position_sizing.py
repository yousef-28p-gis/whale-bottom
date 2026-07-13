"""
Position Sizing: مقارنة 5 نسب من رأس المال
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
entry_signal = (
    whale_spike(whale) &
    (whale_ma(whale, 20) > whale_ma(whale, 50)) &
    (whale_strength(whale, 50) > 50) &
    volume_filter(df) &
    (df['close'] > sma50_daily(df))
)

print(f"🚦 إشارات الدخول: {entry_signal.sum()}")

# تشغيل الباك تست مرة واحدة
r = run_backtest(
    df=df, entry_signal=entry_signal, tp_series=ema21(df),
    atr_series=atr(df,14), sell_series=sell_signal(df),
    swing_mask=swing_lows(df,5), sma50_series=sma50_daily(df),
    tp_mode='ema21', max_hours=48, monthly_limit=0.07, fee=0.001
)

tdf = r['tdf']
print(f"📊 الأساس: {r['trades']} صفقة | WR: {r['wr']:.1f}% | المحفظة: ${r['capital']:.0f}")

# ── محاكاة Position Sizing ──
def simulate_ps(trades, position_pct, monthly_limit_pct=7):
    """
    position_pct: نسبة رأس المال المستخدمة (100 = 100%, 50 = 50%)
    monthly_limit_pct: حد خسارة شهري
    """
    capital = 1000.0
    peak = 1000.0
    max_dd = 0.0
    monthly_pnl = {}

    for _, t in trades.iterrows():
        entry_date = pd.Timestamp(t['entry_time'])
        mk = f"{entry_date.year}-{entry_date.month:02d}"
        ml = monthly_pnl.get(mk, 0.0)

        # حد خسارة شهري
        if ml <= -monthly_limit_pct:
            continue

        # تطبيق نسبة رأس المال
        effective_pnl = (t['pnl_pct'] / 100) * (position_pct / 100)
        capital *= (1 + effective_pnl)

        monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + effective_pnl * 100

        if capital > peak:
            peak = capital
        dd = (capital - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    return capital, max_dd, (capital/1000 - 1)*100

# ── مقارنة النسب ──
print(f"\n{'='*85}")
print(f"🏆 Position Sizing | FET/USDT 15m | {r['trades']} صفقة | WR={r['wr']:.1f}%")
print(f"{'='*85}")
print(f"{'رأس المال/صفقة':<20} {'المحفظة':>12} {'عائد%':>8} {'DD%':>7} {'شارب*':>7} {'كل سنة خضراء؟':<14}")
print(f"{'-'*85}")

for pct in [100, 75, 50, 33, 25]:
    cap, dd, ret = simulate_ps(tdf, pct)

    # هل كل سنة ربحانة؟
    yr_ok = "✅" if cap > 1000 else "❌"
    for yr, grp in tdf.groupby('year'):
        ycap = 1.0
        for _, t in grp.iterrows():
            ycap *= (1 + (t['pnl_pct']/100) * (pct/100))
        if ycap < 1.0:
            yr_ok = f"❌({yr})"
            break

    marker = " ⬅ الحالي" if pct == 100 else ""
    print(f"{pct}%{'':>17} ${cap:>11.0f} {ret:>7.1f}% {dd:>6.1f}% {r['sharpe']:>6.2f} {yr_ok:<14}{marker}")

print(f"\n* Sharpe ثابت لأن توزيع الصفقات نفسه ما تغير")

# ── أفضل سيناريو ──
print(f"\n💡 الخلاصة:")
for pct in [50, 33, 25]:
    cap, dd, ret = simulate_ps(tdf, pct)
    print(f"   {pct}% = DD {dd:.1f}% | ${cap:.0f} (+{ret:.0f}%)")

print(f"\n✅ تم")
