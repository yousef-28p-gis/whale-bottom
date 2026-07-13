"""
DCA + 4hr exit — اختبار على فريمات متعددة
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
df15 = pd.read_csv(DATA_FILE)
df15['timestamp'] = pd.to_datetime(df15['ts'])
df15 = df15.sort_values('timestamp').reset_index(drop=True)

CAP = 1000.0

def resample_ohlcv(df, rule):
    """Resample 15m إلى فريم أعلى"""
    return df.set_index('timestamp').resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()

def run_dca_bars(df_sub, max_bars):
    """DCA backtest مع max_bars بدل max_hours"""
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

    capital = CAP; peak = CAP; max_dd = 0.0
    monthly_pnl = {}; trades = []
    in_trade = False; trade = None

    for i in range(500, n):
        row = df_sub.iloc[i]; ts = row['timestamp']
        mk = f"{ts.year}-{ts.month:02d}"
        ml = monthly_pnl.get(mk, 0.0)
        if ml <= -7 and not in_trade: continue

        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']
                if i < 1 or pd.isna(ema.iloc[i-1]): continue
                tp = ema.iloc[i-1]
                if tp <= ep: continue
                sw_start = max(0, i - 60)
                sw_recent = df_sub.iloc[sw_start:i][sw_mask[sw_start:i]]
                sl = sw_recent['low'].min() * 0.998 if len(sw_recent) > 0 else ep * 0.95
                trade = {'entry_idx': i, 'entry_time': ts, 'entry1': ep, 'entry2': None,
                         'avg_entry': ep, 'allocation': 0.5, 'sl_price': sl, 'tp_price': tp,
                         'dca_done': False}
                in_trade = True
        else:
            if not trade['dca_done']:
                sw2 = max(0, trade['entry_idx'] + 1)
                new_sw = df_sub.iloc[sw2:i+1][sw_mask[sw2:i+1]]
                if len(new_sw) > 0 and new_sw['low'].min() < trade['entry1']:
                    trade['entry2'] = row['close']
                    trade['avg_entry'] = (trade['entry1'] + trade['entry2']) / 2
                    trade['allocation'] = 1.0; trade['dca_done'] = True
                    trade['sl_price'] = new_sw['low'].min() * 0.998

            sw_trail = max(0, i - 100)
            sw_t = df_sub.iloc[sw_trail:i+1][sw_mask[sw_trail:i+1]]
            if len(sw_t) > 0:
                ns = sw_t['low'].min() * 0.998
                if ns > trade['sl_price']: trade['sl_price'] = ns

            exit_reason = None; exit_price = None
            bars_elapsed = i - trade['entry_idx']
            tp_hit = row['high'] >= trade['tp_price']
            sl_hit = (row['high'] >= trade['sl_price']) if trade['sl_price'] > trade['avg_entry'] else (row['low'] <= trade['sl_price'])

            if tp_hit: exit_reason, exit_price = 'TP', trade['tp_price']
            elif i >= 2 and sell.iloc[i-1] >= 60: exit_reason, exit_price = 'SELL', row['close']
            elif sl_hit:
                exit_reason = 'SL_UP' if trade['sl_price'] > trade['avg_entry'] else 'SL'
                exit_price = (min(trade['sl_price'], row['high']) if trade['sl_price'] > trade['avg_entry'] 
                              else max(trade['sl_price'], row['low']))
            elif bars_elapsed >= max_bars: exit_reason, exit_price = 'TIME', row['close']

            if exit_reason:
                pnl_pct = (exit_price - trade['avg_entry']) / trade['avg_entry'] - 0.002
                eff = pnl_pct * trade['allocation']
                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + eff * 100
                capital *= (1 + eff)
                if capital > peak: peak = capital
                dd = (capital - peak) / peak
                if dd < max_dd: max_dd = dd
                trades.append({'pnl_pct': pnl_pct*100, 'exit_reason': exit_reason,
                               'dca_done': trade['dca_done'], 'year': ts.year})
                in_trade = False; trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf)==0: return {'trades':0, 'capital':CAP, 'dd':0, 'wr':0}
    wins = tdf[tdf['pnl_pct']>0]
    wr = len(wins)/len(tdf)*100
    return {'trades':len(tdf), 'wr':wr, 'capital':capital,
            'return_pct':(capital/CAP-1)*100, 'dd':max_dd*100,
            'dca':tdf['dca_done'].sum()}

# ═══════════════════════════════════════════════════
print("⏳ تجهيز الفريمات...")

# Resample
df1h = resample_ohlcv(df15, '1h')
df4h = resample_ohlcv(df15, '4h')

print(f"📊 15m: {len(df15)} شمعة | 1h: {len(df1h)} شمعة | 4h: {len(df4h)} شمعة\n")

# ═══════════════════════════════════════════════════
print(f"{'='*95}")
print(f"🏆 DCA + Swing SL | مقارنة الفريمات")
print(f"{'='*95}")
print(f"{'الفريم':<8} {'حد زمني':<12} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'عائد%':>8} {'DD%':>7} {'DCA':>5}")
print(f"{'-'*95}")

for tf, df_tf, max_bars, label in [
    ('15m', df15, 16, '4 ساعات'),
    ('15m', df15, 48, '12 ساعة'),
    ('15m', df15, 192, '48 ساعة'),
    ('1h', df1h, 4, '4 ساعات'),
    ('1h', df1h, 12, '12 ساعة'),
    ('1h', df1h, 24, '24 ساعة'),
    ('4h', df4h, 3, '12 ساعة'),
    ('4h', df4h, 6, '24 ساعة'),
    ('4h', df4h, 12, '48 ساعة'),
]:
    r = run_dca_bars(df_tf, max_bars)
    if r['trades'] == 0:
        print(f"{tf:<8} {label:<12} {'—':>6} {'—':>6} {'—':>10} {'—':>8} {'—':>7} {'—':>5}")
        continue
    star = ' ⬅' if tf == '15m' and max_bars == 16 else ''
    print(f"{tf:<8} {label:<12} {r['trades']:>6} {r['wr']:>5.1f}% ${r['capital']:>9.0f} {r['return_pct']:>7.1f}% {r['dd']:>6.1f}% {r['dca']:>5}{star}")

print(f"\n✅ تم")
