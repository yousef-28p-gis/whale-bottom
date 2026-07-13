#!/usr/bin/env python3
"""FULL AUDIT: Both strategies with REAL compounding — no sugar coating"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
print(f"📊 {len(df):,} candles | {df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()}", flush=True)

FEE = 0.001; BARS = 200

# Whale
lowest = df['low'].rolling(BARS).min()
at_low = (df['low'] <= lowest).astype(float)
lc = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
sm = lc.ewm(span=3, adjust=False).mean()
hi = sm.rolling(BARS).max()
st = np.where(at_low > 0, (sm + hi * 2) / 3, 0)
df['whale'] = pd.Series(st).ewm(span=3, adjust=False).mean().fillna(0)
df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.02)
df['wma50'] = df['whale'].rolling(50).mean()
df['wma200'] = df['whale'].rolling(200).mean()
df['wstr'] = df['whale'] / df['whale'].rolling(50).max().replace(0, np.nan) * 100
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma'] = df['atr'].rolling(20).mean()
df['vma'] = df['volume'].rolling(20).mean()

# Swings
lb = 5; sh_arr = np.zeros(len(df), dtype=bool); sl_arr = np.zeros(len(df), dtype=bool)
for i in range(lb*2, len(df)):
    w = df['high'].iloc[i-lb*2:i+1]; m = i - lb
    if df['high'].iloc[m] == w.max() and w.values.argmax() == lb: sh_arr[i] = True
    w = df['low'].iloc[i-lb*2:i+1]
    if df['low'].iloc[m] == w.min() and w.values.argmin() == lb: sl_arr[i] = True

def nsl(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if sl_arr[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx] * 0.95

def nsh(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if sh_arr[j]: return df['high'].iloc[j]
    return df['high'].iloc[idx] * 1.05

# RSI + Sell/Buy
delta = df['close'].diff(); g = delta.clip(lower=0); l = -delta.clip(upper=0)
ag = g.ewm(alpha=1/14, adjust=False).mean(); al = l.ewm(alpha=1/14, adjust=False).mean()
df['rsi'] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
vs = df['volume'].rolling(20).mean(); hh20 = df['high'].rolling(20).max().shift(1)
ll20 = df['low'].rolling(20).min().shift(1); ll10 = df['low'].rolling(10).min().shift(1)
hh10 = df['high'].rolling(10).max().shift(1)

# Sell exhaustion
sc = np.zeros(len(df))
sc += ((df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
sc += ((df['high'] > hh20) & (df['close'] < hh20)).astype(int)
sc += ((df['high'] > hh20) & (df['close'] < df['open'])).astype(int)
sc += ((df['close'].shift(1) > df['open'].shift(1)) & (df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
sc += (df['low'] < ll10).astype(int)
sc += ((df['high'] > df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))).astype(int)
df['sell_str'] = sc / 6 * 100

# Buy exhaustion
bc = np.zeros(len(df))
bc += ((df['volume'] > vs * 1.5) & (df['close'] > df['open'])).astype(int)
bc += ((df['low'] < ll20) & (df['close'] > ll20)).astype(int)
bc += ((df['low'] < ll20) & (df['close'] > df['open'])).astype(int)
bc += ((df['close'].shift(1) < df['open'].shift(1)) & (df['volume'] > vs * 1.5) & (df['close'] > df['open'])).astype(int)
bc += (df['high'] > hh10).astype(int)
bc += ((df['low'] < df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))).astype(int)
df['buy_str'] = bc / 6 * 100

# ─── Run a single backtest function ────────────────────────────
def backtest(name, long_only, use_sell_exit, use_reversal, position_pct, monthly_limit_pct):
    """Run backtest with REAL compounding at position_pct% of equity"""
    long_ok = df['wma50'] > df['wma200']
    short_ok = df['wma50'] < df['wma200']
    
    if long_only:
        long_entry = (df['spike'] & (df['wstr'] > 50) & long_ok &
                      (df['volume'] > df['vma']) & (df['atr'] > df['atr_ma']))
        entry_map = {'LONG': np.where(long_entry)[0]}
    else:
        long_entry = (df['spike'] & (df['wstr'] > 50) & long_ok &
                      (df['volume'] > df['vma']) & (df['atr'] > df['atr_ma']))
        short_entry = (df['spike'] & (df['wstr'] > 50) & short_ok &
                       (df['volume'] > df['vma']) & (df['atr'] > df['atr_ma']))
        entry_map = {'LONG': np.where(long_entry)[0], 'SHORT': np.where(short_entry)[0]}
    
    # Merge all entries
    all_entries = []
    for dir_name, idxs in entry_map.items():
        for i in idxs:
            if i >= 500:
                all_entries.append((i, dir_name))
    all_entries.sort()
    
    trades = []
    in_trade = False; exit_done = 0
    equity = 1000
    cmon = df['ts'].iloc[500].month; cyr = df['ts'].iloc[500].year
    month_start = 1000
    eq_curve = [(df['ts'].iloc[500], 1000)]
    
    for ei, dir_name in all_entries:
        if in_trade and ei < exit_done: continue
        
        ts = df['ts'].iloc[ei]
        if ts.month != cmon or ts.year != cyr:
            cmon, cyr = ts.month, ts.year
            month_start = equity
        
        if monthly_limit_pct is not None:
            if (equity - month_start) / month_start * 100 <= -monthly_limit_pct:
                continue
        
        is_long = dir_name == 'LONG'
        entry = df['close'].iloc[ei]
        pos_size = equity * (position_pct / 100)  # amount at risk
        
        if is_long:
            sl = nsl(ei) * 0.998
            if use_reversal:
                tp = 99999  # no TP, exit on SHORT signal
            else:
                tp = entry + df['atr'].iloc[ei] * 3
        else:
            sl = nsh(ei) * 1.002
            tp = entry - df['atr'].iloc[ei] * 3
        
        end = min(ei + 192, len(df))
        result = None; exit_price = entry; exit_idx = ei
        
        for j in range(ei + 1, end):
            if is_long:
                if df['low'].iloc[j] <= sl:
                    result = 'SL'; exit_price = sl; exit_idx = j; break
                if use_sell_exit and df['sell_str'].iloc[j] >= 60:
                    result = 'SELL'; exit_price = df['close'].iloc[j]; exit_idx = j; break
                if use_reversal:
                    short_sig = (df['spike'].iloc[j] and df['wstr'].iloc[j] > 50 and
                                not long_ok.iloc[j] and df['volume'].iloc[j] > df['vma'].iloc[j] and
                                df['atr'].iloc[j] > df['atr_ma'].iloc[j])
                    if short_sig:
                        result = 'REV'; exit_price = df['close'].iloc[j]; exit_idx = j; break
            else:
                if df['high'].iloc[j] >= sl:
                    result = 'SL'; exit_price = sl; exit_idx = j; break
                if df['low'].iloc[j] <= tp:
                    result = 'TP'; exit_price = tp; exit_idx = j; break
                if use_sell_exit and df['buy_str'].iloc[j] >= 60:
                    result = 'BUY'; exit_price = df['close'].iloc[j]; exit_idx = j; break
        
        if result is None:
            result = 'TIME'; exit_price = df['close'].iloc[end-1]; exit_idx = end-1
        
        pnl_pct = (exit_price - entry) / entry * 100 - FEE * 200
        if not is_long:
            pnl_pct = -pnl_pct - FEE * 200  # invert for SHORT
        
        # REAL compounding: equity changes by pnl% of position
        dollar_pnl = pos_size * (pnl_pct / 100)
        equity += dollar_pnl
        
        trades.append({
            'dir': dir_name, 'entry_ts': ts, 'entry_px': entry,
            'exit_ts': df['ts'].iloc[exit_idx], 'exit_px': exit_price,
            'result': result, 'pnl_pct': pnl_pct, 'dollar': dollar_pnl,
        })
        
        eq_curve.append((df['ts'].iloc[exit_idx], equity))
        in_trade = True; exit_done = exit_idx
    
    # Metrics
    n = len(trades)
    if n < 5: return None
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    nw = len(wins); nl = n - nw
    wr = nw / n * 100
    
    total_profit = sum(t['pnl_pct'] for t in wins)
    total_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0))
    net_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]) if nl > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    pnls = [t['pnl_pct'] for t in trades]
    sp = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0
    
    eqs = np.array([e for _, e in eq_curve])
    peak = np.maximum.accumulate(eqs)
    dd = (eqs - peak) / peak * 100
    max_dd = dd.min()
    
    # Consecutive
    from itertools import groupby
    max_win_streak = max((sum(1 for _ in g) for k, g in groupby([t['pnl_pct'] > 0 for t in trades]) if k), default=0)
    max_loss_streak = max((sum(1 for _ in g) for k, g in groupby([t['pnl_pct'] <= 0 for t in trades]) if k), default=0)
    
    lt = [t for t in trades if t['dir'] == 'LONG']
    st = [t for t in trades if t['dir'] == 'SHORT']
    lwr = len([t for t in lt if t['pnl_pct'] > 0]) / len(lt) * 100 if lt else 0
    swr = len([t for t in st if t['pnl_pct'] > 0]) / len(st) * 100 if st else 0
    
    # Monthly
    tdf = pd.DataFrame(trades)
    tdf['month'] = tdf['entry_ts'].dt.to_period('M')
    monthly = tdf.groupby('month')['pnl_pct'].sum()
    
    return {
        'name': name,
        'trades': n, 'wr': wr, 'portfolio': equity,
        'sharpe': sp, 'max_dd': max_dd, 'rr': rr,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'total_profit': total_profit, 'total_loss': total_loss,
        'long_wr': lwr, 'short_wr': swr,
        'long_n': len(lt), 'short_n': len(st),
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'best_month': monthly.max(), 'worst_month': monthly.min(),
    }

# ─── Test both strategies at different position sizes ──────────
print("\n🔍 REAL compounding test...", flush=True)

results = []

# Strategy 1: LONG-only with sell exit (the one we just approved)
for pos in [5, 10, 25, 50, 100]:
    r = backtest(f"LONG-only ({pos}%)", long_only=True, use_sell_exit=True, 
                 use_reversal=False, position_pct=pos, monthly_limit_pct=7)
    if r:
        results.append(r)
        print(f"  {r['name']}: {r['trades']}T | WR:{r['wr']:.0f}% | ${r['portfolio']:,.0f} | DD:{r['max_dd']:.1f}% | S:{r['sharpe']:.1f}", flush=True)

# Strategy 2: LONG+SHORT with reversal for LONG, TP for SHORT
for pos in [5, 10, 25, 50, 100]:
    r = backtest(f"LONG+SHORT ({pos}%)", long_only=False, use_sell_exit=False,
                 use_reversal=True, position_pct=pos, monthly_limit_pct=7)
    if r:
        results.append(r)
        print(f"  {r['name']}: {r['trades']}T | WR:{r['wr']:.0f}% | ${r['portfolio']:,.0f} | DD:{r['max_dd']:.1f}% | S:{r['sharpe']:.1f}", flush=True)

# ─── Print full comparison ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"📊 FULL COMPARISON — REAL COMPOUNDING")
print(f"{'='*70}")

print(f"\n{'Name':<22} {'Trades':>6} {'WR':>5} {'PF':>9} {'DD':>7} {'Sharpe':>6} {'R:R':>5} {'W/L':>7} {'L/S_WR':>12} {'BestM':>7} {'WorstM':>7}")
print("-"*95)

for r in sorted(results, key=lambda x: x['portfolio'], reverse=True):
    wl = f"{r['avg_win']:+.1f}/{r['avg_loss']:+.1f}"
    ls = f"{r['long_wr']:.0f}/{r['short_wr']:.0f}" if r['short_n'] > 0 else f"{r['long_wr']:.0f}/-"
    print(f"{r['name']:<22} {r['trades']:>6} {r['wr']:>4.0f}% ${r['portfolio']:>8,.0f} {r['max_dd']:>6.1f}% {r['sharpe']:>5.1f} {r['rr']:>4.1f}x {wl:>7} {ls:>12} {r['best_month']:>+6.1f}% {r['worst_month']:>+6.1f}%")
