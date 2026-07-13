"""
DCA + Breakeven variants + Fixed SL variants
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
atr_val = atr(df, 14)

n = len(df)
CAP = 1000.0

def run_dca_variants(df, entry_signal, ema, sell, sw_mask, atr_val,
                     max_hours=4, be_pct=None, sl_mode='swing',
                     fixed_sl_pct=None, atr_sl_mult=None):
    """
    be_pct: None (بدون) / رقم (BE بعد ربح %) / 'dynamic50' (BE بعد 50% من مسافة TP) / 'dynamic70'
    sl_mode: 'swing' / 'fixed' / 'atr'
    fixed_sl_pct: نسبة SL ثابتة (مثلاً 1.5 يعني -1.5%)
    atr_sl_mult: مضاعف ATR للـ SL
    """
    capital = CAP
    peak = CAP
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
        if ml <= -7 and not in_trade:
            continue

        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']
                if i < 1 or pd.isna(ema.iloc[i-1]):
                    continue
                tp = ema.iloc[i-1]
                if tp <= ep:
                    continue

                # SL
                if sl_mode == 'swing':
                    sw_start = max(0, i - 60)
                    sw_recent = df.iloc[sw_start:i][sw_mask[sw_start:i]]
                    if len(sw_recent) > 0:
                        sl = sw_recent['low'].min() * 0.998
                    else:
                        sl = ep * 0.95
                elif sl_mode == 'fixed' and fixed_sl_pct:
                    sl = ep * (1 - fixed_sl_pct/100)
                elif sl_mode == 'atr' and atr_sl_mult:
                    if i < 1 or pd.isna(atr_val.iloc[i-1]):
                        sl = ep * 0.95
                    else:
                        sl = ep - atr_val.iloc[i-1] * atr_sl_mult
                else:
                    sl = ep * 0.95

                # BE target if dynamic
                be_target = None
                if be_pct == 'dynamic50':
                    be_target = ep + (tp - ep) * 0.5
                elif be_pct == 'dynamic70':
                    be_target = ep + (tp - ep) * 0.7
                elif be_pct and isinstance(be_pct, (int, float)):
                    be_target = ep * (1 + be_pct/100)

                trade = {
                    'entry_idx': i, 'entry_time': ts,
                    'entry1': ep, 'entry2': None, 'avg_entry': ep,
                    'allocation': 0.5,
                    'sl_price': sl, 'tp_price': tp,
                    'be_target': be_target,
                    'be_activated': False,
                    'dca_done': False, 'dca_idx': None,
                }
                in_trade = True

        else:
            # DCA check
            if not trade['dca_done']:
                sw_start2 = max(0, trade['entry_idx'] + 1)
                new_sw = df.iloc[sw_start2:i+1][sw_mask[sw_start2:i+1]]
                if len(new_sw) > 0:
                    lowest_new = new_sw['low'].min()
                    if lowest_new < trade['entry1']:
                        entry2 = row['close']
                        trade['entry2'] = entry2
                        trade['avg_entry'] = (trade['entry1'] + entry2) / 2
                        trade['allocation'] = 1.0
                        trade['dca_done'] = True
                        trade['dca_idx'] = i
                        # إعادة حساب SL للدخول الجديد
                        if sl_mode == 'swing':
                            trade['sl_price'] = lowest_new * 0.998
                        elif sl_mode == 'fixed' and fixed_sl_pct:
                            trade['sl_price'] = trade['avg_entry'] * (1 - fixed_sl_pct/100)
                        elif sl_mode == 'atr' and atr_sl_mult:
                            trade['sl_price'] = trade['avg_entry'] - atr_val.iloc[i-1] * atr_sl_mult
                        # إعادة حساب BE target
                        if trade['be_target']:
                            if be_pct == 'dynamic50':
                                trade['be_target'] = trade['avg_entry'] + (trade['tp_price'] - trade['avg_entry']) * 0.5
                            elif be_pct == 'dynamic70':
                                trade['be_target'] = trade['avg_entry'] + (trade['tp_price'] - trade['avg_entry']) * 0.7
                            elif isinstance(be_pct, (int, float)):
                                trade['be_target'] = trade['avg_entry'] * (1 + be_pct/100)

            # BE check
            if trade['be_target'] and not trade['be_activated']:
                if row['high'] >= trade['be_target']:
                    trade['be_activated'] = True
                    trade['sl_price'] = trade['avg_entry']

            # Trail SL (swing only)
            if sl_mode == 'swing':
                sw_start_trail = max(0, i - 100)
                sw_trail = df.iloc[sw_start_trail:i+1][sw_mask[sw_start_trail:i+1]]
                if len(sw_trail) > 0:
                    new_sl = sw_trail['low'].min() * 0.998
                    if new_sl > trade['sl_price']:
                        trade['sl_price'] = new_sl

            if trade['be_activated'] and trade['sl_price'] < trade['avg_entry']:
                trade['sl_price'] = trade['avg_entry']

            # Exit
            exit_reason = None
            exit_price = None
            hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600

            tp_hit = row['high'] >= trade['tp_price']
            sl_hit = (row['high'] >= trade['sl_price']) if trade['sl_price'] > trade['avg_entry'] else (row['low'] <= trade['sl_price'])

            if tp_hit:
                exit_reason = 'TP'
                exit_price = trade['tp_price']
            elif i >= 2 and sell.iloc[i-1] >= 60:
                exit_reason = 'SELL'
                exit_price = row['close']
            elif sl_hit:
                if trade['be_activated']:
                    exit_reason = 'BE'
                    exit_price = trade['avg_entry']
                else:
                    exit_reason = 'SL_UP' if trade['sl_price'] > trade['avg_entry'] else 'SL'
                    exit_price = (min(trade['sl_price'], row['high']) if trade['sl_price'] > trade['avg_entry'] 
                                  else max(trade['sl_price'], row['low']))
            elif hours_elapsed >= max_hours:
                exit_reason = 'TIME'
                exit_price = row['close']

            if exit_reason:
                pnl_pct = (exit_price - trade['avg_entry']) / trade['avg_entry'] - 0.002
                effective_pnl = pnl_pct * trade['allocation']
                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + effective_pnl * 100
                capital *= (1 + effective_pnl)
                if capital > peak:
                    peak = capital
                dd = (capital - peak) / peak
                if dd < max_dd:
                    max_dd = dd

                trades.append({
                    'entry_time': trade['entry_time'], 'exit_time': ts,
                    'pnl_pct': pnl_pct * 100, 'effective_pnl_pct': effective_pnl * 100,
                    'exit_reason': exit_reason,
                    'be_activated': trade['be_activated'],
                    'dca_done': trade['dca_done'],
                    'allocation': trade['allocation'] * 100,
                    'duration_m': hours_elapsed * 60, 'year': ts.year,
                })
                in_trade = False
                trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {'trades': 0, 'error': 'no trades'}

    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    wr = len(wins)/len(tdf)*100
    rets = tdf['pnl_pct'].values/100
    sharpe = rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0

    return {
        'trades':len(tdf), 'wr':wr, 'capital':capital,
        'return_pct':(capital/CAP-1)*100, 'dd':max_dd*100, 'sharpe':sharpe,
        'avg_win':wins['pnl_pct'].mean() if len(wins)>0 else 0,
        'avg_loss':losses['pnl_pct'].mean() if len(losses)>0 else 0,
        'be_count':len(tdf[tdf['exit_reason']=='BE']),
        'dca_count':tdf['dca_done'].sum(),
        'tdf':tdf,
    }

# ═══════════════════════════════════════════════════
print(f"📦 {len(df)} شمعة | 🚦 {entry_signal.sum()} إشارة\n")

# Part 1: BE variants
print("="*95)
print("🏆 الجزء ١: Breakeven variants | DCA + 4hr + Swing SL")
print("="*95)

be_configs = [
    ('بدون BE', None),
    ('BE ثابت +2%', 2.0),
    ('BE ثابت +3%', 3.0),
    ('BE ديناميك 50% من TP', 'dynamic50'),
    ('BE ديناميك 70% من TP', 'dynamic70'),
]

for name, be in be_configs:
    r = run_dca_variants(df, entry_signal, ema, sell, sw_mask, atr_val, 4, be_pct=be)
    if 'error' in r:
        print(f"🔒 {name}: ❌ {r['error']}")
        continue
    exits = r['tdf']['exit_reason'].value_counts().to_dict()
    print(f"🔒 {name}: {r['trades']}T | WR={r['wr']:.1f}% | ${r['capital']:.0f} (+{r['return_pct']:.0f}%) | DD={r['dd']:.1f}% | Sharpe={r['sharpe']:.2f}")
    print(f"   BE={r['be_count']} | DCA={r['dca_count']} | AvgW={r['avg_win']:.2f}% AvgL={r['avg_loss']:.2f}%")
    print(f"   مخارج: {exits}")

# Part 2: Fixed SL variants
print(f"\n{'='*95}")
print(f"🏆 الجزء ٢: Fixed SL variants | DCA + 4hr + بدون BE")
print(f"{'='*95}")

sl_configs = [
    ('Swing SL (الأساس)', 'swing', None, None),
    ('SL ثابت -1%', 'fixed', 1.0, None),
    ('SL ثابت -2%', 'fixed', 2.0, None),
    ('SL ثابت -3%', 'fixed', 3.0, None),
    ('SL = 1x ATR', 'atr', None, 1.0),
    ('SL = 1.5x ATR', 'atr', None, 1.5),
    ('SL = 2x ATR', 'atr', None, 2.0),
]

for name, sm, fp, am in sl_configs:
    r = run_dca_variants(df, entry_signal, ema, sell, sw_mask, atr_val, 4,
                         sl_mode=sm, fixed_sl_pct=fp, atr_sl_mult=am)
    if 'error' in r:
        print(f"🔒 {name}: ❌ {r['error']}")
        continue
    exits = r['tdf']['exit_reason'].value_counts().to_dict()
    star = ' ⬅' if sm == 'swing' else ''
    print(f"🔒 {name}: {r['trades']}T | WR={r['wr']:.1f}% | ${r['capital']:.0f} (+{r['return_pct']:.0f}%) | DD={r['dd']:.1f}% | Sharpe={r['sharpe']:.2f}{star}")
    print(f"   DCA={r['dca_count']} | AvgW={r['avg_win']:.2f}% AvgL={r['avg_loss']:.2f}%")
    print(f"   مخارج: {exits}")

print(f"\n✅ تم")
