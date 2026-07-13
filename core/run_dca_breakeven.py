"""
DCA + Breakeven: تحريك SL لنقطة الدخول بعد ربح 0.5%، 1%، 1.5%
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

n = len(df)
CAPITAL_START = 1000.0

def run_dca_be(df, entry_signal, ema, sell, sw_mask, max_hours=4, be_pct=None):
    """
    be_pct: نسبة الربح اللي بعدها نحرك SL لنقطة التعادل (None = بدون)
    """
    capital = CAPITAL_START
    peak = CAPITAL_START
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

                sw_start = max(0, i - 60)
                sw_recent = df.iloc[sw_start:i][sw_mask[sw_start:i]]
                if len(sw_recent) > 0:
                    sl = sw_recent['low'].min() * 0.998
                else:
                    sl = ep * 0.95

                trade = {
                    'entry_idx': i, 'entry_time': ts,
                    'entry1': ep, 'entry2': None, 'avg_entry': ep,
                    'allocation': 0.5,
                    'sl_price': sl, 'tp_price': tp,
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
                        trade['sl_price'] = lowest_new * 0.998

            # Breakeven check: إذا وصل السعر لربح be_pct%
            if be_pct and not trade['be_activated']:
                be_price = trade['avg_entry'] * (1 + be_pct/100)
                if row['high'] >= be_price:
                    trade['be_activated'] = True
                    trade['sl_price'] = trade['avg_entry']  # حرك SL للدخول

            # Trail SL (بس إذا breakeven بعد ما تفعّل — ما ننزل SL تحت الدخول)
            sw_start_trail = max(0, i - 100)
            sw_trail = df.iloc[sw_start_trail:i+1][sw_mask[sw_start_trail:i+1]]
            if len(sw_trail) > 0:
                new_sl = sw_trail['low'].min() * 0.998
                if new_sl > trade['sl_price']:
                    trade['sl_price'] = new_sl

            # لا نسمح لـ SL ينزل تحت avg_entry إذا BE مفعّل
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
                    exit_price = min(trade['sl_price'], row['high']) if trade['sl_price'] > trade['avg_entry'] else max(trade['sl_price'], row['low'])
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
                    'avg_entry': trade['avg_entry'], 'exit_price': exit_price,
                    'be_activated': trade['be_activated'],
                    'dca_done': trade['dca_done'],
                    'allocation': trade['allocation'] * 100,
                    'duration_m': hours_elapsed * 60,
                    'year': ts.year,
                })
                in_trade = False
                trade = None

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        return {'trades': 0}

    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    be_trades = tdf[tdf['exit_reason'] == 'BE']
    wr = len(wins)/len(tdf)*100
    rets = tdf['pnl_pct'].values/100
    sharpe = rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0

    return {
        'trades':len(tdf), 'wr':wr, 'capital':capital,
        'return_pct':(capital/CAPITAL_START-1)*100, 'dd':max_dd*100, 'sharpe':sharpe,
        'avg_win':wins['pnl_pct'].mean() if len(wins)>0 else 0,
        'avg_loss':losses['pnl_pct'].mean() if len(losses)>0 else 0,
        'be_count':len(be_trades),
        'be_saved': len(be_trades),  # صفقات كانت حتخسر وصارت BE
        'dca_count':tdf['dca_done'].sum(),
        'tdf':tdf,
    }

# ── المقارنة ──
print(f"📦 {len(df)} شمعة | 🚦 إشارات: {entry_signal.sum()}\n")
print(f"{'='*95}")
print(f"🏆 DCA + Breakeven | حد زمني 4 ساعات | FET/USDT 15m")
print(f"{'='*95}")

for be_pct, label in [(None, 'بدون BE'), (0.5, 'BE بعد +0.5%'), (1.0, 'BE بعد +1%'), (1.5, 'BE بعد +1.5%')]:
    r = run_dca_be(df, entry_signal, ema, sell, sw_mask, max_hours=4, be_pct=be_pct)
    if r['trades']==0:
        print(f"🔒 {label}: ❌ 0")
        continue
    tdf = r['tdf']
    exits = tdf['exit_reason'].value_counts().to_dict()
    
    star = ' ⬅ الأفضل' if r['capital'] == max(
        [x['capital'] for x in 
         [run_dca_be(df, entry_signal, ema, sell, sw_mask, 4, b) for b in [None,0.5,1.0,1.5]]
         if x['trades']>0]
    ) else ''
    
    print(f"\n🔒 {label}: {r['trades']}T | WR={r['wr']:.1f}% | ${r['capital']:.0f} (+{r['return_pct']:.0f}%) | DD={r['dd']:.1f}% | Sharpe={r['sharpe']:.2f}{star}")
    print(f"   BE={r['be_count']} | تعزيز={r['dca_count']} | AvgW={r['avg_win']:.2f}% AvgL={r['avg_loss']:.2f}%")
    print(f"   مخارج: TP={exits.get('TP',0)} SL={exits.get('SL',0)}+{exits.get('SL_UP',0)} SELL={exits.get('SELL',0)} TIME={exits.get('TIME',0)} BE={exits.get('BE',0)}")
    
    # توزيع BE — هل أصلاً كانوا خاسرين؟
    be_df = tdf[tdf['exit_reason']=='BE']
    if len(be_df) > 0:
        print(f"   📊 صفقات BE: {len(be_df)} | متوسط المدة={be_df['duration_m'].mean():.0f}د")

print(f"\n✅ تم")
