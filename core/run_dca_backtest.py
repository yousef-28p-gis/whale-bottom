"""
استراتيجية التعزيز: 50% دخول + 50% عند قاع سوينج جديد
"""
import sys
sys.path.insert(0, '/data/trading28')

import pandas as pd
import numpy as np
from core.indicators import (
    whale_indicator, whale_ma, whale_strength, whale_spike,
    volume_filter, sma50_daily, ema21, atr, sell_signal, swing_lows
)

DATA_FILE = '/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv'
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📦 {len(df)} شمعة | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# ── المؤشرات ──
whale = whale_indicator(df, 200)
entry_signal = (
    whale_spike(whale) &
    (whale_ma(whale, 20) > whale_ma(whale, 50)) &
    (whale_strength(whale, 50) > 50) &
    volume_filter(df) &
    (df['close'] > sma50_daily(df))
)
ema = ema21(df)
sell = sell_signal(df)
sw_mask = swing_lows(df, 5)

# ═══════════════════════════════════════════════════
# باك تست مع التعزيز (DCA)
# ═══════════════════════════════════════════════════
n = len(df)
CAPITAL_START = 1000.0

def run_dca_backtest(use_dca=True):
    capital = CAPITAL_START
    equity_peak = CAPITAL_START
    max_dd = 0.0
    monthly_pnl = {}
    trades = []
    in_trade = False
    has_dca = False
    trade = None

    for i in range(500, n):
        row = df.iloc[i]
        ts = row['timestamp']
        mk = f"{ts.year}-{ts.month:02d}"
        ml = monthly_pnl.get(mk, 0.0)

        if ml <= -7 and not in_trade:
            continue

        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']

                # TP = EMA21 من البار السابق
                if i < 1 or pd.isna(ema.iloc[i-1]):
                    continue
                tp = ema.iloc[i-1]
                if tp <= ep:
                    continue

                # SL = أقرب قاع سوينج
                sw_start = max(0, i - 60)
                sw_recent = df.iloc[sw_start:i][sw_mask[sw_start:i]]
                if len(sw_recent) > 0:
                    sl = sw_recent['low'].min() * 0.998
                else:
                    sl = ep * 0.95

                allocation = 0.5 if use_dca else 1.0
                trade = {
                    'entry_idx': i, 'entry_time': ts,
                    'entry1': ep, 'entry2': None,
                    'avg_entry': ep,
                    'allocation': allocation,  # نسبة من رأس المال الأصلي
                    'sl_price': sl,
                    'tp_price': tp,  # ثابت = EMA21 عند الدخول
                    'highest_close': ep,
                    'dca_done': False,
                    'dca_idx': None,
                }
                in_trade = True
                has_dca = False

        else:
            if row['close'] > trade['highest_close']:
                trade['highest_close'] = row['close']

            # ── فحص التعزيز: قاع سوينج جديد ظهر ──
            if use_dca and not trade['dca_done']:
                # هل ظهر قاع سوينج جديد أقل من السعر الحالي؟
                sw_start2 = max(0, trade['entry_idx'] + 1)
                new_swings = df.iloc[sw_start2:i+1][sw_mask[sw_start2:i+1]]
                if len(new_swings) > 0:
                    lowest_new = new_swings['low'].min()
                    if lowest_new < trade['entry1']:
                        # تعزيز!
                        entry2 = row['close']
                        trade['entry2'] = entry2
                        trade['avg_entry'] = (trade['entry1'] + entry2) / 2
                        trade['allocation'] = 1.0  # صار 100%
                        trade['dca_done'] = True
                        trade['dca_idx'] = i
                        # SL جديد = أدنى قاع سوينج بعد التعزيز
                        trade['sl_price'] = lowest_new * 0.998

            # تحديث SL (بعد التعزيز أو بدونه)
            sw_start_trail = max(0, i - 100)
            sw_trail = df.iloc[sw_start_trail:i+1][sw_mask[sw_start_trail:i+1]]
            if len(sw_trail) > 0:
                new_sl = sw_trail['low'].min() * 0.998
                if new_sl > trade['sl_price']:
                    trade['sl_price'] = new_sl

            # ── فحص الخروج ──
            exit_reason = None
            exit_price = None

            tp_hit = row['high'] >= trade['tp_price']

            if trade['sl_price'] > trade['avg_entry']:
                sl_hit = row['high'] >= trade['sl_price']
            else:
                sl_hit = row['low'] <= trade['sl_price']

            if tp_hit:
                exit_reason = 'TP'
                exit_price = trade['tp_price']
            elif i >= 2 and sell.iloc[i-1] >= 60:
                exit_reason = 'SELL'
                exit_price = row['close']
            elif sl_hit:
                if trade['sl_price'] > trade['avg_entry']:
                    exit_reason = 'SL_UP'
                    exit_price = min(trade['sl_price'], row['high'])
                else:
                    exit_reason = 'SL'
                    exit_price = max(trade['sl_price'], row['low'])

            if not exit_reason:
                hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600
                if hours_elapsed >= 48:
                    exit_reason = 'TIME'
                    exit_price = row['close']

            if exit_reason:
                pnl_pct = (exit_price - trade['avg_entry']) / trade['avg_entry'] - 0.002
                effective_pnl = pnl_pct * trade['allocation']

                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + effective_pnl * 100
                capital *= (1 + effective_pnl)

                if capital > equity_peak:
                    equity_peak = capital
                dd = (capital - equity_peak) / equity_peak
                if dd < max_dd:
                    max_dd = dd

                trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': ts,
                    'entry_price': trade['entry1'],
                    'entry2': trade['entry2'],
                    'avg_entry': trade['avg_entry'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100,
                    'effective_pnl_pct': effective_pnl * 100,
                    'exit_reason': exit_reason,
                    'tp_price': trade['tp_price'],
                    'sl_price': trade['sl_price'],
                    'dca_done': trade['dca_done'],
                    'dca_idx': trade['dca_idx'],
                    'allocation': trade['allocation'] * 100,
                    'duration_m': (ts - trade['entry_time']).total_seconds() / 60,
                    'year': ts.year,
                })
                in_trade = False
                trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {'trades': 0}

    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    wr = len(wins) / len(tdf) * 100
    rets = tdf['pnl_pct'].values / 100
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0

    dca_count = tdf['dca_done'].sum()
    dca_trades = tdf[tdf['dca_done']]
    dca_wr = len(dca_trades[dca_trades['pnl_pct'] > 0]) / len(dca_trades) * 100 if len(dca_trades) > 0 else 0

    return {
        'trades': len(tdf), 'wins': len(wins), 'losses': len(losses),
        'wr': wr, 'capital': capital,
        'return_pct': (capital / CAPITAL_START - 1) * 100,
        'dd': max_dd * 100, 'sharpe': sharpe,
        'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'dca_count': dca_count, 'dca_wr': dca_wr,
        'tdf': tdf,
    }

# ── تشغيل ──
configs = [
    ('1️⃣ الحالي (100% دفعة وحدة)', run_dca_backtest(use_dca=False)),
    ('2️⃣ 50% + تعزيز 50% عند قاع سوينج', run_dca_backtest(use_dca=True)),
]

for name, r in configs:
    print(f"\n{'─'*70}")
    print(f"🔄 {name}")
    print(f"{'─'*70}")

    if r['trades'] == 0:
        print("❌ صفر صفقات")
        continue

    tdf = r['tdf']
    print(f"   صفقات: {r['trades']} | WR: {r['wr']:.1f}%")
    if 'dca_count' in r:
        print(f"   منها بتعزيز: {r['dca_count']} | WR مع تعزيز: {r['dca_wr']:.0f}%")
    print(f"   متوسط ربح: +{r['avg_win']:.2f}% | متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"   محفظة: $1000 → ${r['capital']:.2f} (+{r['return_pct']:.1f}%)")
    print(f"   DD: {r['dd']:.1f}% | Sharpe: {r['sharpe']:.2f} | مدة: {tdf['duration_m'].mean():.0f}د")

    if 'dca_count' in r and r['dca_count'] > 0:
        exits = tdf['exit_reason'].value_counts().to_dict()
        print(f"   مخارج: TP={exits.get('TP',0)} SL={exits.get('SL',0)}+{exits.get('SL_UP',0)} SELL={exits.get('SELL',0)} TIME={exits.get('TIME',0)}")

    print(f"   سنوي:", end="")
    for yr, grp in tdf.groupby('year'):
        ycap = 1.0
        for _, t in grp.iterrows():
            ycap *= (1 + (t.get('effective_pnl_pct', t['pnl_pct']) / 100))
        yret = (ycap - 1) * 100
        ywr = (grp['pnl_pct'] > 0).sum() / len(grp) * 100
        print(f" {yr}:{len(grp)}T/{ywr:.0f}%/{yret:+.0f}%", end="")
    print()

print(f"\n{'='*75}")
print(f"🏆 المقارنة")
print(f"{'='*75}")
for name, r in configs:
    if r['trades'] == 0:
        continue
    dca_info = f" | تعزيز={r.get('dca_count','?')}" if 'dca_count' in r else ""
    print(f"{name}: ${r['capital']:.0f} (+{r['return_pct']:.1f}%) | DD: {r['dd']:.1f}% | WR: {r['wr']:.1f}%{dca_info}")

print(f"\n✅ تم")
