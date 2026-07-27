#!/usr/bin/env python3
"""Quick test: RSI<30+PrevRed on 120-day data"""
import json, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

with open(f'{DATA_DIR}/daily_120d.json') as f:
    all_data = json.load(f)

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

def backtest(tp, sl, max_hold=7):
    all_signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close'])
        volume = np.array(data['volume'])
        n = len(close)
        if n < 80: continue
        
        # RSI(14) + pct change
        close_s = pd.Series(close)
        delta = close_s.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        rsi = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
        pct = close_s.pct_change() * 100
        
        for i in range(50, n - 2):
            if rsi.iloc[i] >= 30 or np.isnan(rsi.iloc[i]): continue
            if pct.iloc[i] >= 0: continue  # must be red day
            
            # Confirmation: next candle green
            if data['close'][i+1] <= data['open'][i+1]: continue
            
            entry_p = data['close'][i+1]
            all_signals.append({'coin': coin, 'idx': i+1, 'entry': entry_p, 'date': data['ts'][i+1]})
    
    all_signals.sort(key=lambda s: s['date'])
    trades = []
    capital = INITIAL_CAPITAL
    active = {}
    
    for sig in all_signals:
        coin = sig['coin']; ei = sig['idx']
        if coin in active and active[coin] > ei: continue
        
        data = all_data[coin]
        c = np.array(data['close']); h = np.array(data['high']); l = np.array(data['low'])
        n = len(c)
        
        tp_p = sig['entry'] * (1 + tp); sl_p = sig['entry'] * (1 - sl)
        ep = None; et = None; ex = None
        
        for j in range(ei + 1, min(ei + max_hold, n)):
            if l[j] <= sl_p: ep = sl_p; et = 'SL'; ex = j; break
            elif h[j] >= tp_p: ep = tp_p; et = 'TP'; ex = j; break
        
        if ep is None:
            end = min(ei + max_hold, n-1)
            ep = c[end]; et = 'TIME'; ex = end
        
        pnl = (ep / sig['entry'] - 1) * 100 - COMMISSION * 100
        sz = capital * 0.10
        capital += sz * pnl / 100
        trades.append({'pnl': pnl, 'type': et, 'cap': capital})
        active[coin] = ex
        active = {k: v for k, v in active.items() if v > ei}
    
    return trades, capital

print(f"🔬 RSI<30+PrevRed on 120-day data ({len(all_data)} coins)")
print(f"{'='*80}")

for tp, sl, label in [(0.05, 0.025, "TP5/SL2.5"), (0.10, 0.05, "TP10/SL5"), (0.15, 0.06, "TP15/SL6")]:
    trades, final = backtest(tp, sl)
    if not trades:
        print(f"  {label}: 0 trades")
        continue
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]; losses = df[df['pnl'] <= 0]
    wr = len(wins) / len(df) * 100
    eq = np.array([1000] + [t['cap'] for t in trades])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    ret = (final / 1000 - 1) * 100
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 else 999
    tp_hits = len(df[df['type'] == 'TP'])
    
    print(f"  {label}: {len(df):>4d} trades | WR {wr:.1f}% | Return {ret:+.1f}% | DD {dd.min():.2f}% | PF {pf:.2f} | TP:{tp_hits} | Avg {df['pnl'].mean():+.2f}%")

print(f"\n✅ Done!")
