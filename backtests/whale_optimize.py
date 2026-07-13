#!/usr/bin/env python3
"""Whale Strategy Optimization — grid search over key parameters"""
import pandas as pd, numpy as np, os, sys, time
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

CACHE = '/data/trading28/backtests/cache'
cache_file = f"{CACHE}/FET_USDT_15m.csv"
df = pd.read_csv(cache_file, parse_dates=['ts'])
print(f"📊 {len(df)} candles | {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}", flush=True)

# ─── Whale Indicator ────────────────────────────────────────────
print("🐋 Computing whale indicator...", flush=True)
lowest_30 = df['low'].rolling(30).min()
df['at_low'] = (df['low'] <= lowest_30).astype(float)
low_change = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
smooth = low_change.ewm(span=3, adjust=False).mean()
highest = smooth.rolling(30).max()
strength = np.where(df['at_low'] > 0, (smooth + highest * 2) / 3, 0)
df['whale'] = pd.Series(strength).ewm(span=3, adjust=False).mean().fillna(0)
df['whale_spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.02)

# Whale MAs
df['w_ma50'] = df['whale'].rolling(50).mean()
df['w_ma200'] = df['whale'].rolling(200).mean()
df['w_peak50'] = df['whale'].rolling(50).max()
df['w_strength'] = df['whale'] / df['w_peak50'].replace(0, np.nan) * 100
df['trend_up'] = df['w_ma50'] > df['w_ma200']

# Price indicators
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma20'] = df['atr'].rolling(20).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()

# 6-condition sell/buy exhaustion
vs20 = df['volume'].rolling(20).mean()
hh20 = df['high'].rolling(20).max().shift(1)
ll20 = df['low'].rolling(20).min().shift(1)
ll10 = df['low'].rolling(10).min().shift(1)
hh10 = df['high'].rolling(10).max().shift(1)

# RSI
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

# Sell exhaustion (strength 0-100%)
sell_cnt = np.zeros(len(df))
sell_cnt += ((df['volume'] > vs20 * 1.5) & (df['close'] < df['open'])).astype(int)
sell_cnt += ((df['high'] > hh20) & (df['close'] < hh20)).astype(int)
sell_cnt += ((df['high'] > hh20) & (df['close'] < df['open'])).astype(int)
sell_cnt += ((df['close'].shift(1) > df['open'].shift(1)) & (df['volume'] > vs20 * 1.5) & (df['close'] < df['open'])).astype(int)
sell_cnt += (df['low'] < ll10).astype(int)
sell_cnt += ((df['high'] > df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))).astype(int)
df['sell_strength'] = sell_cnt / 6.0 * 100

buy_cnt = np.zeros(len(df))
buy_cnt += ((df['volume'] > vs20 * 1.5) & (df['close'] > df['open'])).astype(int)
buy_cnt += ((df['low'] < ll20) & (df['close'] > ll20)).astype(int)
buy_cnt += ((df['low'] < ll20) & (df['close'] > df['open'])).astype(int)
buy_cnt += ((df['close'].shift(1) < df['open'].shift(1)) & (df['volume'] > vs20 * 1.5) & (df['close'] > df['open'])).astype(int)
buy_cnt += (df['high'] > hh10).astype(int)
buy_cnt += ((df['low'] < df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))).astype(int)
df['buy_strength'] = buy_cnt / 6.0 * 100

# Swing detection (5-bar)
lookback = 5
swing_h = np.zeros(len(df), dtype=bool)
swing_l = np.zeros(len(df), dtype=bool)
for i in range(lookback*2, len(df)):
    win_h = df['high'].iloc[i-lookback*2:i+1]
    mid = i - lookback
    if df['high'].iloc[mid] == win_h.max() and win_h.values.argmax() == lookback:
        swing_h[i] = True
    win_l = df['low'].iloc[i-lookback*2:i+1]
    if df['low'].iloc[mid] == win_l.min() and win_l.values.argmin() == lookback:
        swing_l[i] = True

def nearest_swing_low(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if swing_l[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx] * 0.95

def nearest_swing_high(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if swing_h[j]: return df['high'].iloc[j]
    return df['high'].iloc[idx] * 1.05

# ─── Grid Search ────────────────────────────────────────────────
print("🔍 Running grid search...", flush=True)

FEE = 0.001
CAPITAL = 1000  # full capital per trade

configs = []
# Base config
for w_str in [50, 60, 70, 80]:
    for vol_mult in [1.0, 1.5]:
        for monthly_limit in [5, 7, 10, 99]:  # 99 = no limit
            for use_signal_exit in [False, True]:
                for long_reversal in [False, True]:
                    # Skip combos that don't make sense
                    if use_signal_exit and long_reversal:
                        continue  # can't have both signal exit and reversal
                    
                    configs.append({
                        'whale_strength': w_str,
                        'vol_mult': vol_mult,
                        'monthly_limit': monthly_limit if monthly_limit < 99 else None,
                        'use_signal_exit': use_signal_exit,
                        'long_reversal': long_reversal,
                    })

print(f"  📊 {len(configs)} combinations to test", flush=True)

results = []
best_trades = None
best_config = None
best_portfolio = -999

for ci, cfg in enumerate(configs):
    ws = cfg['whale_strength']
    vm = cfg['vol_mult']
    ml = cfg['monthly_limit']
    se = cfg['use_signal_exit']
    lr = cfg['long_reversal']
    
    # Entry signals
    long_entry = (df['whale_spike'] & 
                  (df['w_strength'] > ws) & 
                  df['trend_up'] &
                  (df['volume'] > df['vol_ma20'] * vm) &
                  (df['atr'] > df['atr_ma20']))
    
    short_entry = (df['whale_spike'] & 
                   (df['w_strength'] > ws) & 
                   ~df['trend_up'] &
                   (df['volume'] > df['vol_ma20'] * vm) &
                   (df['atr'] > df['atr_ma20']))
    
    entry_idxs = np.where(long_entry | short_entry)[0]
    
    if len(entry_idxs) == 0:
        results.append({**cfg, 'trades': 0, 'portfolio': 1000, 'wr': 0})
        continue
    
    # Simulate
    trades = []
    in_trade = False
    trade_exit_idx = 0
    current_month = df['ts'].iloc[200].month
    current_year = df['ts'].iloc[200].year
    month_start_eq = CAPITAL
    equity = CAPITAL
    direction = 0  # 1=LONG, -1=SHORT
    
    for ei in entry_idxs:
        if ei < 200: continue
        if in_trade and ei < trade_exit_idx: continue
        
        ts = df['ts'].iloc[ei]
        em, ey = ts.month, ts.year
        if em != current_month or ey != current_year:
            current_month, current_year = em, ey
            month_start_eq = equity
        
        if ml is not None:
            monthly_loss = (equity - month_start_eq) / month_start_eq * 100
            if monthly_loss < -ml: continue
        
        is_long = long_entry.iloc[ei]
        entry_price = df['close'].iloc[ei]
        
        if is_long:
            sl = nearest_swing_low(ei) * 0.998
            if lr:
                tp = 99999  # no TP — exit on SHORT signal only
            elif se:
                tp = entry_price + df['atr'].iloc[ei] * 3
            else:
                tp = entry_price + df['atr'].iloc[ei] * 3
        else:
            sl = nearest_swing_high(ei) * 1.002
            tp = entry_price - df['atr'].iloc[ei] * 3
        
        # Simulate trade
        max_hold = 48 * 4  # 48h on 15m = 192 bars
        end_idx = min(ei + max_hold, len(df))
        result = None
        exit_price = entry_price
        exit_idx = ei
        
        for j in range(ei + 1, end_idx):
            if is_long:
                if df['low'].iloc[j] <= sl:
                    result = 'SL'; exit_price = sl; exit_idx = j; break
                if tp < 90000 and df['high'].iloc[j] >= tp:
                    result = 'TP'; exit_price = tp; exit_idx = j; break
                if se and df['sell_strength'].iloc[j] >= 80:
                    result = 'SIG'; exit_price = df['close'].iloc[j]; exit_idx = j; break
                if lr:
                    # Check for SHORT signal (reversal)
                    short_sig = (short_entry.iloc[j] and 
                                df['w_strength'].iloc[j] > ws and
                                df['volume'].iloc[j] > df['vol_ma20'].iloc[j] * vm and
                                df['atr'].iloc[j] > df['atr_ma20'].iloc[j])
                    if short_sig:
                        result = 'REV'; exit_price = df['close'].iloc[j]; exit_idx = j; break
            else:
                if df['high'].iloc[j] >= sl:
                    result = 'SL'; exit_price = sl; exit_idx = j; break
                if df['low'].iloc[j] <= tp:
                    result = 'TP'; exit_price = tp; exit_idx = j; break
                if se and df['buy_strength'].iloc[j] >= 80:
                    result = 'SIG'; exit_price = df['close'].iloc[j]; exit_idx = j; break
        
        if result is None:
            result = 'TIME'; exit_price = df['close'].iloc[end_idx-1]; exit_idx = end_idx-1
        
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        if is_long:
            pnl_pct -= FEE * 200
        else:
            pnl_pct = -pnl_pct - FEE * 200
        
        trades.append({'is_long': is_long, 'result': result, 'pnl_pct': pnl_pct,
                       'entry_idx': ei, 'exit_idx': exit_idx})
        
        in_trade = True
        trade_exit_idx = exit_idx
        equity += CAPITAL * (pnl_pct / 100)  # full capital compounding
    
    n = len(trades)
    if n == 0:
        results.append({**cfg, 'trades': 0, 'portfolio': 1000, 'wr': 0})
        continue
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    wr = len(wins) / n * 100
    net = sum(t['pnl_pct'] for t in trades)
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]) if (n - len(wins)) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Sharpe
    pnls = [t['pnl_pct'] for t in trades]
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
    
    # Max DD
    eq = CAPITAL
    eq_curve = [eq]
    for t in trades:
        eq += CAPITAL * (t['pnl_pct'] / 100)
        eq_curve.append(eq)
    peak = np.maximum.accumulate(eq_curve)
    dd = (np.array(eq_curve) - peak) / peak * 100
    max_dd = dd.min()
    
    # LONG/SHORT breakdown
    long_trades = [t for t in trades if t['is_long']]
    short_trades = [t for t in trades if not t['is_long']]
    long_wr = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades) * 100 if long_trades else 0
    short_wr = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades) * 100 if short_trades else 0
    
    portfolio = eq
    
    results.append({**cfg, 'trades': n, 'portfolio': portfolio, 'wr': wr, 
                    'sharpe': sharpe, 'max_dd': max_dd, 'net_pnl': net,
                    'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
                    'long_wr': long_wr, 'short_wr': short_wr,
                    'long_n': len(long_trades), 'short_n': len(short_trades)})
    
    if portfolio > best_portfolio:
        best_portfolio = portfolio
        best_config = cfg
        best_trades = trades
    
    if (ci + 1) % 20 == 0:
        print(f"  [{ci+1}/{len(configs)}] best so far: ${best_portfolio:,.0f} ({ws}%/{vm}x/{ml}%/{'SE' if se else ''}{'LR' if lr else ''})", flush=True)

# ─── Results ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"🐋 WHALE STRATEGY OPTIMIZATION RESULTS")
print(f"{'='*60}")

# Sort by portfolio
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('portfolio', ascending=False)

# Show top 10
print(f"\n🏆 TOP 10 CONFIGS:\n")
print(f"{'#':<3} {'Str%':<5} {'Vol×':<5} {'ML%':<5} {'SigX':<5} {'LR':<4} {'Trades':<7} {'PF':>8} {'WR':>6} {'L/S_WR':>12} {'Sharpe':>7} {'DD%':>6} {'R:R':>5}")
print("-" * 85)

for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    ws = row['whale_strength']
    vm = row['vol_mult']
    ml = row['monthly_limit']
    se = '✓' if row['use_signal_exit'] else '✗'
    lr = '✓' if row['long_reversal'] else '✗'
    ls_wr = f"{row['long_wr']:.0f}/{row['short_wr']:.0f}"
    print(f"{rank:<3} {ws:<5.0f} {vm:<5.1f} {ml if ml else '-':<5} {se:<5} {lr:<4} {row['trades']:<7} ${row['portfolio']:>7,.0f} {row['wr']:>5.1f}% {ls_wr:>12} {row['sharpe']:>6.2f} {row['max_dd']:>5.1f}% {row['rr']:>4.1f}x")

# Baseline comparison
print(f"\n📊 BASELINE (70%/1.5x/7%):")
base = results_df[(results_df['whale_strength']==70) & (results_df['vol_mult']==1.5) & 
                  (results_df['monthly_limit']==7) & (~results_df['use_signal_exit']) & (~results_df['long_reversal'])]
if len(base) > 0:
    b = base.iloc[0]
    print(f"  Trades: {b['trades']} | PF: ${b['portfolio']:,.0f} | WR: {b['wr']:.1f}% | Sharpe: {b['sharpe']:.2f} | DD: {b['max_dd']:.1f}%")
else:
    print("  Not found in results")

print(f"\n✅ Done! {len(results_df)} configs tested. Best: ${best_portfolio:,.0f}")
