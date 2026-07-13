"""
DCA + 4hr — اختبار على فترات زمنية مختلفة
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

CAP = 1000.0

def run_dca_period(df_sub, label=""):
    whale = whale_indicator(df_sub, 200)
    entry_signal = (
        whale_spike(whale) &
        (whale_ma(whale, 20) > whale_ma(whale, 50)) &
        (whale_strength(whale, 50) > 50) &
        volume_filter(df_sub) &
        (df_sub['close'] > sma50_daily(df_sub))
    )
    ema = ema21(df_sub)
    sell = sell_signal(df_sub)
    sw_mask = swing_lows(df_sub, 5)
    n = len(df_sub)

    capital = CAP
    peak = CAP
    max_dd = 0.0
    monthly_pnl = {}
    trades = []
    in_trade = False
    trade = None

    for i in range(500, n):
        row = df_sub.iloc[i]
        ts = row['timestamp']
        mk = f"{ts.year}-{ts.month:02d}"
        ml = monthly_pnl.get(mk, 0.0)
        if ml <= -7 and not in_trade:
            continue

        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']
                if i < 1 or pd.isna(ema.iloc[i-1]): continue
                tp = ema.iloc[i-1]
                if tp <= ep: continue

                sw_start = max(0, i - 60)
                sw_recent = df_sub.iloc[sw_start:i][sw_mask[sw_start:i]]
                sl = sw_recent['low'].min() * 0.998 if len(sw_recent) > 0 else ep * 0.95

                trade = {
                    'entry_idx': i, 'entry_time': ts,
                    'entry1': ep, 'entry2': None, 'avg_entry': ep,
                    'allocation': 0.5,
                    'sl_price': sl, 'tp_price': tp,
                    'dca_done': False,
                }
                in_trade = True

        else:
            if not trade['dca_done']:
                sw_start2 = max(0, trade['entry_idx'] + 1)
                new_sw = df_sub.iloc[sw_start2:i+1][sw_mask[sw_start2:i+1]]
                if len(new_sw) > 0 and new_sw['low'].min() < trade['entry1']:
                    entry2 = row['close']
                    trade['entry2'] = entry2
                    trade['avg_entry'] = (trade['entry1'] + entry2) / 2
                    trade['allocation'] = 1.0
                    trade['dca_done'] = True
                    trade['sl_price'] = new_sw['low'].min() * 0.998

            sw_start_trail = max(0, i - 100)
            sw_trail = df_sub.iloc[sw_start_trail:i+1][sw_mask[sw_start_trail:i+1]]
            if len(sw_trail) > 0:
                new_sl = sw_trail['low'].min() * 0.998
                if new_sl > trade['sl_price']:
                    trade['sl_price'] = new_sl

            exit_reason = None
            exit_price = None
            hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600

            tp_hit = row['high'] >= trade['tp_price']
            sl_hit = (row['high'] >= trade['sl_price']) if trade['sl_price'] > trade['avg_entry'] else (row['low'] <= trade['sl_price'])

            if tp_hit:
                exit_reason = 'TP'; exit_price = trade['tp_price']
            elif i >= 2 and sell.iloc[i-1] >= 60:
                exit_reason = 'SELL'; exit_price = row['close']
            elif sl_hit:
                exit_reason = 'SL_UP' if trade['sl_price'] > trade['avg_entry'] else 'SL'
                exit_price = (min(trade['sl_price'], row['high']) if trade['sl_price'] > trade['avg_entry'] 
                              else max(trade['sl_price'], row['low']))
            elif hours_elapsed >= 4:
                exit_reason = 'TIME'; exit_price = row['close']

            if exit_reason:
                pnl_pct = (exit_price - trade['avg_entry']) / trade['avg_entry'] - 0.002
                effective_pnl = pnl_pct * trade['allocation']
                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + effective_pnl * 100
                capital *= (1 + effective_pnl)
                if capital > peak: peak = capital
                dd = (capital - peak) / peak
                if dd < max_dd: max_dd = dd
                trades.append({
                    'pnl_pct': pnl_pct * 100, 'effective_pnl_pct': effective_pnl * 100,
                    'exit_reason': exit_reason, 'dca_done': trade['dca_done'],
                    'duration_m': hours_elapsed * 60, 'year': ts.year,
                })
                in_trade = False
                trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {'trades': 0, 'capital': CAP, 'dd': 0, 'wr': 0, 'error': 'no trades'}

    wins = tdf[tdf['pnl_pct'] > 0]
    wr = len(wins)/len(tdf)*100
    return {
        'trades': len(tdf), 'wr': wr, 'capital': capital,
        'return_pct': (capital/CAP - 1)*100,
        'dd': max_dd*100,
        'dca_count': tdf['dca_done'].sum(),
    }

# ── تقسيم الفترة ──
full_start = df['timestamp'].iloc[0]
full_end = df['timestamp'].iloc[-1]

periods = [
    ('كاملة (2019→2026)', full_start, full_end),
    ('آخر سنة (2025-07→2026-07)', '2025-07-01', '2026-07-09'),
    ('آخر سنتين (2024-07→2026-07)', '2024-07-01', '2026-07-09'),
    ('آخر 3 سنوات (2023-07→2026-07)', '2023-07-01', '2026-07-09'),
    ('أول 3 سنوات (2019→2022)', '2019-01-01', '2022-12-31'),
    ('سنة 2021 (صاعدة)', '2021-01-01', '2021-12-31'),
    ('سنة 2022 (هابطة)', '2022-01-01', '2022-12-31'),
    ('سنة 2024', '2024-01-01', '2024-12-31'),
    ('سنة 2025', '2025-01-01', '2025-12-31'),
    ('أول 6 شهور 2026', '2026-01-01', '2026-07-09'),
]

print(f"{'='*85}")
print(f"🏆 DCA + 4hr + Swing SL | اختبار على فترات مختلفة")
print(f"{'='*85}")
print(f"{'الفترة':<30} {'من':>12} {'إلى':>12} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'DD%':>7}")
print(f"{'-'*85}")

for label, start, end in periods:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df_sub = df[mask].reset_index(drop=True)
    if len(df_sub) < 1000:
        print(f"{label:<30} {'—':>12} {'—':>12} {'—':>6} {'—':>6} {'—':>10} {'—':>7} (بيانات غير كافية)")
        continue

    r = run_dca_period(df_sub, label)
    if 'error' in r:
        print(f"{label:<30} {str(df_sub['timestamp'].iloc[0])[:10]:>12} {str(df_sub['timestamp'].iloc[-1])[:10]:>12} {r['trades']:>6} {'—':>6} {'—':>10} {'—':>7}")
        continue

    dd_mark = '⚠️' if r['dd'] < -20 else ('✅' if r['dd'] > -10 else '')
    ret_mark = '🔥' if r['return_pct'] > 50 else ('✅' if r['return_pct'] > 0 else '❌')
    print(f"{label:<30} {str(df_sub['timestamp'].iloc[0])[:10]:>12} {str(df_sub['timestamp'].iloc[-1])[:10]:>12} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>9.0f} {r['dd']:>6.1f}% {dd_mark}{ret_mark}")

print(f"\n✅ تم")
