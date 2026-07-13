"""
مقارنة: استغلال الومضة للخروج
  ١. الإعداد الحالي: EMA21 فقط
  ٢. ومضة تصريف كخروج بديل (بدون EMA21)
  ٣. ومضة تصريف + EMA21 (أيهما أولاً)
  ٤. ومضة تصريف + EMA21 + SELL (أقربهم)
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

# ── المؤشرات الأساسية ──
print("⏳ حساب المؤشرات...")

whale = whale_indicator(df, 200)

# ⚡ ومضة تصريف (عند القمم) — صورة طبق الأصل من whale_indicator لكن على highs
lookback = 200
highest_n = df['high'].rolling(lookback).max()
at_high = df['high'] >= highest_n
high_change = abs(df['high'] - df['high'].shift(1)) / df['high'] * 100
smooth_change_h = high_change.ewm(span=3, adjust=False).mean()
highest_change_h = smooth_change_h.rolling(lookback).max()
strength_h = np.where(at_high, (smooth_change_h + highest_change_h * 2) / 3, 0)
dump_raw = pd.Series(strength_h, index=df.index).ewm(span=3, adjust=False).mean().fillna(0)

# Spike للتصريف (ارتداد من الصفر)
dump_spike = (dump_raw > dump_raw.shift(1)) & (dump_raw.shift(1) <= 0.02)
# قوة التصريف كنسبة من الذروة
dump_strength = pd.Series(
    np.where(dump_raw.rolling(50).max() > 0, dump_raw / dump_raw.rolling(50).max() * 100, 0),
    index=df.index
)
# USE: dump_spike & (dump_strength > 50)  كإشارة خروج

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

# إشارة الدخول (موحدة للكل)
entry_signal = (
    spike &
    (wma20 > wma50) &
    (strength > 50) &
    vol_ok &
    (df['close'] > sma50)
)

print(f"🚦 إشارات الدخول: {entry_signal.sum()}")
print(f"🔄 ومضات التصريف: {dump_spike.sum()} (قوية>50%: {(dump_spike & (dump_strength > 50)).sum()})")

# ═══════════════════════════════════════════════════
# نسخة معدلة من backtest_engine تدعم dump exit
# ═══════════════════════════════════════════════════

def run_with_dump(df, entry_signal, tp_series, atr_series, sell_series,
                  swing_mask, sma50_series, dump_spike, dump_strength,
                  tp_mode='ema21', use_dump=False, dump_exit_only=False,
                  max_hours=48, monthly_limit=0.07, fee=0.001):
    """
    use_dump=True: الخروج على EMA21 ⚡أو⚡ ومضة تصريف>50% (أيهما أولاً)
    dump_exit_only=True: الخروج على ومضة التصريف فقط (بدون EMA21)
    """
    n = len(df)
    capital = 1000.0
    equity_peak = 1000.0
    max_dd = 0.0
    monthly_pnl = {}
    trades = []
    in_trade = False
    trade = None

    for i in range(500, n):
        row = df.iloc[i]
        ts = row['timestamp']
        mk = f"{ts.year}-{ts.month:02d}"

        ml = monthly_pnl.get(mk, 0.0)
        if ml <= -monthly_limit and not in_trade:
            continue

        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']

                if not dump_exit_only:
                    if tp_mode == 'ema21':
                        if i < 1 or pd.isna(tp_series.iloc[i-1]):
                            continue
                        tp = tp_series.iloc[i-1]
                        if tp <= ep:
                            continue
                    else:
                        if i < 1 or pd.isna(atr_series.iloc[i-1]):
                            continue
                        tp = ep + 3 * atr_series.iloc[i-1]
                else:
                    tp = ep * 2  # هدف بعيد جداً (لن يلمس)، نعتمد على dump exit

                # SL
                sw_start = max(0, i - 60)
                sw_recent = df.iloc[sw_start:i][swing_mask[sw_start:i]]
                if len(sw_recent) > 0:
                    sl = sw_recent['low'].min() * 0.998
                else:
                    sl = ep * 0.95

                trade = {
                    'entry_idx': i, 'entry_time': ts,
                    'entry_price': ep, 'sl_price': sl,
                    'tp_price': tp, 'highest_close': ep,
                }
                in_trade = True

        else:
            if row['close'] > trade['highest_close']:
                trade['highest_close'] = row['close']

            # تحديث SL
            sw_start = max(0, i - 100)
            sw_recent = df.iloc[sw_start:i+1][swing_mask[sw_start:i+1]]
            if len(sw_recent) > 0:
                new_sl = sw_recent['low'].min() * 0.998
                if new_sl > trade['sl_price']:
                    trade['sl_price'] = new_sl

            exit_reason = None
            exit_price = None

            tp_hit = row['high'] >= trade['tp_price']

            if trade['sl_price'] > trade['entry_price']:
                sl_hit = row['high'] >= trade['sl_price']
            else:
                sl_hit = row['low'] <= trade['sl_price']

            # ومضة تصريف قوية (shift(1) — منع look-ahead)
            dump_exit = False
            if use_dump or dump_exit_only:
                if i >= 2:
                    dump_strong = dump_spike.iloc[i-1] and dump_strength.iloc[i-1] > 50
                    if dump_strong:
                        dump_exit = True

            # أولوية الخروج
            if tp_hit:
                exit_reason = 'TP'
                exit_price = trade['tp_price']
            elif dump_exit:
                exit_reason = 'DUMP'
                exit_price = row['close']
            elif i >= 2 and not tp_hit and sell_series.iloc[i-1] >= 60:
                exit_reason = 'SELL'
                exit_price = row['close']
            elif sl_hit:
                if trade['sl_price'] > trade['entry_price']:
                    exit_reason = 'SL_UP'
                    exit_price = min(trade['sl_price'], row['high'])
                else:
                    exit_reason = 'SL'
                    exit_price = max(trade['sl_price'], row['low'])

            # TIME
            if not exit_reason:
                hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600
                if hours_elapsed >= max_hours:
                    exit_reason = 'TIME'
                    exit_price = row['close']

            if exit_reason:
                pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price'] - 2 * fee
                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + pnl_pct
                capital *= (1 + pnl_pct)

                if capital > equity_peak:
                    equity_peak = capital
                dd = (capital - equity_peak) / equity_peak
                if dd < max_dd:
                    max_dd = dd

                trades.append({
                    'entry_time': trade['entry_time'], 'exit_time': ts,
                    'entry_idx': trade['entry_idx'], 'exit_idx': i,
                    'entry_price': trade['entry_price'], 'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100, 'exit_reason': exit_reason,
                    'tp_price': trade['tp_price'], 'sl_price': trade['sl_price'],
                    'sl_above_entry': trade['sl_price'] > trade['entry_price'],
                    'highest_close': trade['highest_close'],
                    'duration_m': (ts - trade['entry_time']).total_seconds() / 60,
                    'year': ts.year,
                })
                in_trade = False
                trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {'trades': 0, 'error': 'No trades'}

    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    wr = len(wins) / len(tdf) * 100
    rets = tdf['pnl_pct'].values / 100
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0

    return {
        'trades': len(tdf), 'wins': len(wins), 'losses': len(losses),
        'wr': wr, 'capital': capital,
        'return_pct': (capital / 1000 - 1) * 100,
        'dd': max_dd * 100, 'sharpe': sharpe,
        'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'max_win': tdf['pnl_pct'].max(), 'max_loss': tdf['pnl_pct'].min(),
        'avg_dur': tdf['duration_m'].mean(), 'tdf': tdf,
    }

# ═══════════════════════════════════════════════════
# تشغيل السيناريوهات الأربعة
# ═══════════════════════════════════════════════════

configs = [
    {
        'name': '1️⃣ الحالي: EMA21 فقط',
        'use_dump': False,
        'dump_exit_only': False,
    },
    {
        'name': '2️⃣ ومضة تصريف فقط (بدون EMA21)',
        'use_dump': False,
        'dump_exit_only': True,
    },
    {
        'name': '3️⃣ ومضة تصريف + EMA21 (أيهما أولاً)',
        'use_dump': True,
        'dump_exit_only': False,
    },
]

results = {}

for cfg in configs:
    print(f"\n{'─'*70}")
    print(f"🔄 {cfg['name']}")
    print(f"{'─'*70}")

    r = run_with_dump(
        df=df, entry_signal=entry_signal,
        tp_series=ema, atr_series=atr_val, sell_series=sell,
        swing_mask=sw_mask, sma50_series=sma50,
        dump_spike=dump_spike, dump_strength=dump_strength,
        tp_mode='ema21',
        use_dump=cfg['use_dump'],
        dump_exit_only=cfg['dump_exit_only'],
        max_hours=48, monthly_limit=0.07, fee=0.001,
    )

    if 'error' in r:
        print(f"❌ {r['error']}")
        continue

    results[cfg['name']] = r
    tdf = r['tdf']

    print(f"   صفقات: {r['trades']} | WR: {r['wr']:.1f}%")
    print(f"   متوسط ربح: +{r['avg_win']:.2f}% | متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"   محفظة: $1000 → ${r['capital']:.2f} (+{r['return_pct']:.1f}%)")
    print(f"   DD: {r['dd']:.1f}% | Sharpe: {r['sharpe']:.2f} | مدة: {r['avg_dur']:.0f}د")
    exits = tdf['exit_reason'].value_counts().to_dict()
    print(f"   مخارج: TP={exits.get('TP',0)} | DUMP={exits.get('DUMP',0)} | SL={exits.get('SL',0)}+{exits.get('SL_UP',0)} | SELL={exits.get('SELL',0)} | TIME={exits.get('TIME',0)}")

    # DUMP exit stats if any
    dump_exits = tdf[tdf['exit_reason'] == 'DUMP']
    if len(dump_exits) > 0:
        print(f"   📊 صفقات DUMP: {len(dump_exits)} | WR={len(dump_exits[dump_exits['pnl_pct']>0])/len(dump_exits)*100:.0f}% | متوسط={dump_exits['pnl_pct'].mean():.2f}%")

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

# ═══════════════════════════════════════════════════
# جدول المقارنة
# ═══════════════════════════════════════════════════
print(f"\n{'='*95}")
print(f"🏆 مقارنة: استغلال الومضة للخروج | FET/USDT 15m | LONG only")
print(f"{'='*95}")
print(f"{'الإعداد':<40} {'صفقات':>5} {'WR%':>6} {'المحفظة':>12} {'عائد%':>9} {'DD%':>7} {'شارب':>6}")
print(f"{'-'*95}")
for name, r in results.items():
    print(f"{name:<40} {r['trades']:>5} {r['wr']:>5.1f}% ${r['capital']:>11.2f} {r['return_pct']:>8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

print(f"\n✅ تم")
