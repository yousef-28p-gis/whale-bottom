"""
═══════════════════════════════════════════════════════════════
🔬 اختبار شامل — كل مؤشر، كل إعداد، كل خطوة
═══════════════════════════════════════════════════════════════
"""
import sys
sys.path.insert(0, '/data/trading28')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("🔬 إعادة اختبار كل شي من الصفر")
print("=" * 70)

# ============================================================
# ١. بيانات وهمية لفحص المؤشرات
# ============================================================
print("\n" + "─" * 70)
print("❶ فحص المؤشرات ببيانات وهمية")
print("─" * 70)

np.random.seed(42)
n_test = 500
dates = [datetime(2024,1,1) + timedelta(minutes=15*i) for i in range(n_test)]
price = 100.0
prices = []
for i in range(n_test):
    price *= (1 + np.random.normal(0.0002, 0.02))
    prices.append(price)

test_df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
    'close': prices,
    'volume': np.random.uniform(100, 10000, n_test),
})

from core.indicators import (
    whale_indicator, whale_ma, whale_strength, whale_spike,
    volume_filter, sma50_daily, ema21, atr, sell_signal, swing_lows
)

# فحص كل مؤشر
checks = []

# Whale
w = whale_indicator(test_df, 200)
checks.append(("حوت >= 0", (w >= 0).all()))
checks.append(("حوت NaN", w.isna().sum() == 0))

# WMA
w20 = whale_ma(w, 20)
checks.append(("WMA20 NaN بعد 20", w20.iloc[30:].isna().sum() == 0))

# Strength
ws = whale_strength(w, 50)
checks.append(("قوة 0-100", ((ws >= 0) & (ws <= 100) | ws.isna()).all()))

# Spike
sp = whale_spike(w)
checks.append(("Spike boolean", sp.dtype == bool))

# Volume
vf = volume_filter(test_df)
checks.append(("حجم boolean", vf.dtype == bool))

# SMA50 daily
s50 = sma50_daily(test_df)
checks.append(("SMA50 NaN أول 50 يوم", s50.iloc[:100].isna().any()))  # طبيعي

# EMA21
e21 = ema21(test_df)
checks.append(("EMA21 NaN", e21.isna().sum() < 50))

# ATR
a = atr(test_df, 14)
checks.append(("ATR > 0", (a.dropna() > 0).all()))

# Sell signal
ss = sell_signal(test_df)
checks.append(("بيع 0-100", ((ss >= 0) & (ss <= 100)).all()))

# Swing lows
sl = swing_lows(test_df, 5)
checks.append(("Swing boolean", sl.dtype == bool))
checks.append(("Swing فيه قيم", sl.sum() > 0))

all_ok = True
for name, ok in checks:
    status = "✅" if ok else "❌"
    if not ok: all_ok = False
    print(f"   {status} {name}")

if all_ok:
    print(f"   ✅ كل المؤشرات سليمة")
else:
    print(f"   ❌ فيه مشاكل في المؤشرات!")
    sys.exit(1)

# ============================================================
# ٢. فحص look-ahead في المؤشرات
# ============================================================
print(f"\n{'─'*70}")
print(f"❷ فحص Look-Ahead")
print(f"{'─'*70}")

# فكرة: نحذف آخر شمعة ونشوف إذا المؤشرات تغيرت للشمعة قبل الأخيرة
df1 = test_df.iloc[:-1].copy()
df2 = test_df.copy()

# Whale
w1 = whale_indicator(df1, 200)
w2 = whale_indicator(df2, 200)
# القيم للشمعة الأخيرة في df1 يجب تطابق القيم في df2 لنفس الشمعة
match = (w1.iloc[-1] == w2.iloc[len(df1)-1])
print(f"   حوت look-ahead: {'✅ لا يوجد' if match else '⚠️ يوجد تغير'} (آخر قيمة: {w1.iloc[-1]:.4f} vs {w2.iloc[len(df1)-1]:.4f})")

# EMA21
e1 = ema21(df1)
e2 = ema21(df2)
match_e = (e1.iloc[-1] == e2.iloc[len(df1)-1])
print(f"   EMA21 look-ahead: {'✅ لا يوجد' if match_e else '⚠️ يوجد تغير'}")

# ============================================================
# ٣. تحميل بيانات حقيقية
# ============================================================
print(f"\n{'─'*70}")
print(f"❸ تحميل بيانات FET")
print(f"{'─'*70}")

DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"   {len(df)} شمعة | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"   NaN: close={df['close'].isna().sum()} high={df['high'].isna().sum()} low={df['low'].isna().sum()}")

# ============================================================
# ٤. حساب المؤشرات على بيانات FET
# ============================================================
print(f"\n{'─'*70}")
print(f"❹ مؤشرات FET")
print(f"{'─'*70}")

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

print(f"   🐋 حوت > 0: {(whale > 0).sum():,} / {len(whale):,} = {(whale > 0).mean()*100:.1f}%")
print(f"   📈 wMA20 > wMA50: {(wma20 > wma50).sum():,} = {(wma20 > wma50).mean()*100:.1f}%")
print(f"   💪 قوة > 50%: {(strength > 50).sum():,}")
print(f"   🚀 حوت spike: {spike.sum():,}")
print(f"   📊 حجم > 1.5x: {vol_ok.sum():,} = {vol_ok.mean()*100:.1f}%")
print(f"   📈 سعر > SMA50: {(df['close'] > sma50).sum():,} = {(df['close'] > sma50).mean()*100:.1f}%")
print(f"   🎯 EMA21 > سعر: {(ema > df['close']).sum():,}")
print(f"   📉 إشارة بيع ≥60%: {(sell >= 60).sum():,}")
print(f"   🔄 قيعان سوينج: {sw_mask.sum():,}")

# ============================================================
# ٥. إشارات الدخول — تحليل كل شرط
# ============================================================
print(f"\n{'─'*70}")
print(f"❺ تحليل شروط الدخول")
print(f"{'─'*70}")

entry_signal = spike & (wma20 > wma50) & (strength > 50) & vol_ok & (df['close'] > sma50)
print(f"   🎯 إجمالي الإشارات: {entry_signal.sum()}")

# حلل كل إشارة
sig_indices = entry_signal[entry_signal].index
sig_count = len(sig_indices)

# نسبة كل شرط من الإشارات الضائعة
would_be = spike & (wma20 > wma50) & (strength > 50) & vol_ok  # بدون فلتر السوق
print(f"   ⚠️ بدون فلتر SMA50: {would_be.sum()} (+{would_be.sum() - sig_count})")

# توزيع سنوي
print(f"   📅 توزيع سنوي:")
for yr in range(2019, 2027):
    mask = df['timestamp'].dt.year == yr
    cnt = entry_signal[mask].sum()
    bar = '█' * cnt
    print(f"     {yr}: {cnt:>4d} {bar}")

# ============================================================
# ٦. اختبار Grid Search للمعلمات الأساسية
# ============================================================
print(f"\n{'─'*70}")
print(f"❻ Grid Search — المعلمات الأساسية")
print(f"{'─'*70}")

from core.backtest_engine import run_backtest

# اختبار قوة الحوت: 30%, 40%, 50%, 60%, 70%
print(f"\n   📊 اختبار عتبة القوة:")
for pct in [30, 40, 50, 60, 70]:
    sig = spike & (wma20 > wma50) & (strength > pct) & vol_ok & (df['close'] > sma50)
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
    if 'error' not in r:
        print(f"     قوة > {pct}%: {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")

# اختبار حجم التداول: 1.0x, 1.5x, 2.0x
print(f"\n   📊 اختبار مضاعف الحجم:")
for mult in [1.0, 1.5, 2.0]:
    vf = df['volume'] > df['volume'].rolling(20).mean() * mult
    sig = spike & (wma20 > wma50) & (strength > 50) & vf & (df['close'] > sma50)
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
    if 'error' not in r:
        print(f"     حجم > {mult}x: {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")

# اختبار الحد الشهري: 5%, 7%, 10%, no limit
print(f"\n   📊 اختبار الحد الشهري:")
for lim in [0.05, 0.07, 0.10, 0.99]:
    sig = spike & (wma20 > wma50) & (strength > 50) & vol_ok & (df['close'] > sma50)
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, lim, 0.001)
    if 'error' not in r:
        label = f"{lim*100:.0f}%" if lim < 0.99 else "بدون"
        print(f"     حد {label}: {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")

# ============================================================
# ٧. مقارنة EMA21 vs 3ATR vs بدون هدف
# ============================================================
print(f"\n{'─'*70}")
print(f"❼ مقارنة أنواع الأهداف")
print(f"{'─'*70}")

sig = spike & (wma20 > wma50) & (strength > 50) & vol_ok & (df['close'] > sma50)

for tp_mode, label in [('ema21', 'EMA21'), ('3atr', '3ATR')]:
    r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, tp_mode, 48, 0.07, 0.001)
    if 'error' not in r:
        print(f"   {label}: {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}% | Sharpe={r['sharpe']:.2f}")

# ============================================================
# ٨. اختبار بدون فلتر SMA50
# ============================================================
print(f"\n{'─'*70}")
print(f"❽ اختبار بدون فلتر SMA50")
print(f"{'─'*70}")

sig_no_sma = spike & (wma20 > wma50) & (strength > 50) & vol_ok
for tp_mode in ['ema21', '3atr']:
    r = run_backtest(df, sig_no_sma, ema, atr_val, sell, sw_mask, sma50, tp_mode, 48, 0.07, 0.001)
    if 'error' not in r:
        print(f"   {tp_mode}: {r['trades']:>4d}T | WR={r['wr']:.0f}% | ${r['capital']:.0f} | DD={r['dd']:.1f}%")

# ============================================================
# ٩. آخر ٣ شهور
# ============================================================
print(f"\n{'─'*70}")
print(f"❾ آخر ٣ شهور (أبريل-يوليو ٢٠٢٦)")
print(f"{'─'*70}")

cutoff = df['timestamp'].max() - pd.Timedelta(days=90)
recent_mask = df['timestamp'] >= cutoff
sig_recent = sig & recent_mask
print(f"   إشارات: {sig_recent.sum()}")

# نشغل باك تست على كامل البيانات لكن نعرض صفقات آخر ٣ شهور فقط
r = run_backtest(df, sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)
if 'error' not in r:
    recent_trades = r['tdf'][r['tdf']['entry_time'] >= cutoff]
    if len(recent_trades) > 0:
        print(f"   صفقات: {len(recent_trades)}")
        for _, t in recent_trades.iterrows():
            emoji = '🟢' if t['pnl_pct'] > 0 else '🔴'
            print(f"     {emoji} {t['entry_time'].strftime('%m/%d %H:%M')} | {t['exit_reason']:5s} | {t['pnl_pct']:+.2f}% | {t['duration_m']:.0f}د")
    else:
        print(f"   ❌ لا صفقات في آخر ٣ شهور")
        print(f"   السعر: ${df['close'].iloc[-1]:.4f} | SMA50: ${sma50.iloc[-1]:.4f}")

# ============================================================
# ١٠. الملخص النهائي
# ============================================================
print(f"\n{'='*70}")
print(f"🏁 الملخص النهائي")
print(f"{'='*70}")

# أفضل إعدادات
best_sig = spike & (wma20 > wma50) & (strength > 50) & vol_ok & (df['close'] > sma50)
best_r = run_backtest(df, best_sig, ema, atr_val, sell, sw_mask, sma50, 'ema21', 48, 0.07, 0.001)

# تحقق نهائي
from core.verify import verify_trades
final_ok = verify_trades(df, best_r)

if final_ok and 'error' not in best_r:
    tdf = best_r['tdf']
    print(f"\n   📋 أفضل إعداد:")
    print(f"   حوت: 200 بار | قوة > 50% | حجم > 1.5x | سعر > SMA50 اليومي")
    print(f"   هدف: EMA21 | SL: دعم متحرك | حد شهري: 7% | 48h")
    print(f"")
    print(f"   📊 النتيجة النهائية:")
    print(f"   صفقات: {best_r['trades']} | WR: {best_r['wr']:.1f}%")
    print(f"   محفظة: $1000 → ${best_r['capital']:.0f} (+{best_r['return_pct']:.1f}%)")
    print(f"   DD: {best_r['dd']:.1f}% | Sharpe: {best_r['sharpe']:.2f}")
    print(f"   متوسط ربح: +{best_r['avg_win']:.2f}% | متوسط خسارة: {best_r['avg_loss']:.2f}%")
    print(f"   أقصى ربح: +{best_r['max_win']:.2f}% | أقصى خسارة: {best_r['max_loss']:.2f}%")
    print(f"")
    print(f"   📅 سنوي:")
    for yr, grp in tdf.groupby('year'):
        ycap = 1.0
        for _, t in grp.iterrows():
            ycap *= (1 + t['pnl_pct']/100)
        yret = (ycap-1)*100
        ywr = (grp['pnl_pct']>0).sum()/len(grp)*100
        print(f"   {yr}: {len(grp):>3d}T | WR={ywr:.0f}% | {'+' if yret>=0 else ''}{yret:.1f}% {'✅' if yret>=0 else '❌'}")
    
    # إحصائيات إضافية
    print(f"\n   📊 توزيع النتائج:")
    for (lo, hi), label in [((-100, -5), '<-5%'), ((-5, -3), '-5~-3%'), ((-3, -1), '-3~-1%'), ((-1, 0), '-1~0%'), ((0, 1), '0~1%'), ((1, 3), '1~3%'), ((3, 5), '3~5%'), ((5, 100), '>5%')]:
        cnt = len(tdf[(tdf['pnl_pct'] >= lo) & (tdf['pnl_pct'] < hi)])
        bar = '█' * (cnt // 5)
        print(f"   {label:>8s}: {cnt:>4d} {bar}")
    
    print(f"\n   🏆 أقصى سلسلة ربح: ", end='')
    max_w = cur = 0
    for _, t in tdf.iterrows():
        if t['pnl_pct'] > 0: cur += 1; max_w = max(max_w, cur)
        else: cur = 0
    print(f"{max_w}")
    
    print(f"   📉 أقصى سلسلة خسارة: ", end='')
    max_l = cur = 0
    for _, t in tdf.iterrows():
        if t['pnl_pct'] <= 0: cur += 1; max_l = max(max_l, cur)
        else: cur = 0
    print(f"{max_l}")
    
else:
    print(f"   ❌ فشل التحقق النهائي!")
