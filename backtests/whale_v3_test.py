#!/usr/bin/env python3
"""Whale v3: 100-bar whale + MA200 price filter + LONG reversal only"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m.csv', parse_dates=['ts'])
print(f"📊 {len(df)} candles", flush=True)

# ─── Whale v3: 100-bar lookback ─────────────────────────────────
print("🐋 Computing whale (100-bar)...", flush=True)
lowest_100 = df['low'].rolling(100).min()
at_low = (df['low'] <= lowest_100).astype(float)
low_change = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
smooth = low_change.ewm(span=3, adjust=False).mean()
highest_100 = smooth.rolling(100).max()
strength = np.where(at_low > 0, (smooth + highest_100 * 2) / 3, 0)
df['whale'] = pd.Series(strength).ewm(span=3, adjust=False).mean().fillna(0)
df['whale_spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.02)

# Whale MAs
df['w_ma50'] = df['whale'].rolling(50).mean()
df['w_ma200'] = df['whale'].rolling(200).mean()
df['w_peak50'] = df['whale'].rolling(50).max()
df['w_strength'] = df['whale'] / df['w_peak50'].replace(0, np.nan) * 100

# Price MAs
df['p_ma200'] = df['close'].rolling(200).mean()

# Other indicators
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['atr_ma20'] = df['atr'].rolling(20).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()

# Swings for SL
lb = 5
swing_h = np.zeros(len(df), dtype=bool)
swing_l = np.zeros(len(df), dtype=bool)
for i in range(lb*2, len(df)):
    w = df['high'].iloc[i-lb*2:i+1]; m = i-lb
    if df['high'].iloc[m]==w.max() and w.values.argmax()==lb: swing_h[i]=True
    w = df['low'].iloc[i-lb*2:i+1]
    if df['low'].iloc[m]==w.min() and w.values.argmin()==lb: swing_l[i]=True

def nsl(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if swing_l[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx]*0.95

def nsh(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if swing_h[j]: return df['high'].iloc[j]
    return df['high'].iloc[idx]*1.05

print(f"🐋 Whale spikes: {df['whale_spike'].sum()}", flush=True)

# ─── Configs to test ────────────────────────────────────────────
configs = []

# Grid: price_ma200 filter × whale_strength × vol_mult
for use_pma in [False, True]:
    for ws in [50, 60, 70]:
        for vm in [1.0, 1.5]:
            configs.append({
                'name': f"{'PMA200' if use_pma else 'NoPMA'}_{ws}%_{vm}x",
                'use_pma': use_pma, 'whale_strength': ws, 'vol_mult': vm,
            })

FEE = 0.001
CAPITAL = 1000

print(f"🔍 Testing {len(configs)} configs...", flush=True)

results = []
for ci, cfg in enumerate(configs):
    up = cfg['use_pma']
    ws = cfg['whale_strength']
    vm = cfg['vol_mult']
    
    # Direction rules
    if up:
        # Price above MA200 → LONG + SHORT both allowed
        # Price below MA200 → SHORT only
        price_bull = df['p_ma200'].notna() & (df['close'] > df['p_ma200'])
        price_bear = df['p_ma200'].notna() & (df['close'] < df['p_ma200'])
        
        long_ok = (df['w_ma50'] > df['w_ma200']) & price_bull
        short_ok = (df['w_ma50'] < df['w_ma200']) & (price_bear | price_bull)
    else:
        long_ok = df['w_ma50'] > df['w_ma200']
        short_ok = df['w_ma50'] < df['w_ma200']
    
    long_entry = (df['whale_spike'] & (df['w_strength'] > ws) & long_ok &
                  (df['volume'] > df['vol_ma20'] * vm) & (df['atr'] > df['atr_ma20']))
    short_entry = (df['whale_spike'] & (df['w_strength'] > ws) & short_ok &
                   (df['volume'] > df['vol_ma20'] * vm) & (df['atr'] > df['atr_ma20']))
    
    entry_idxs = np.where(long_entry | short_entry)[0]
    
    if len(entry_idxs) == 0:
        results.append({**cfg, 'trades': 0, 'portfolio': 1000, 'wr': 0})
        continue
    
    # Simulate
    trades = []
    in_trade = False
    exit_idx_done = 0
    equity = CAPITAL
    cmon = df['ts'].iloc[200].month
    cyr = df['ts'].iloc[200].year
    mstart = CAPITAL
    
    for ei in entry_idxs:
        if ei < 300: continue
        if in_trade and ei < exit_idx_done: continue
        
        ts = df['ts'].iloc[ei]
        if ts.month != cmon or ts.year != cyr:
            cmon, cyr = ts.month, ts.year
            mstart = equity
        
        # Monthly 7% limit
        if (equity - mstart) / mstart * 100 < -7:
            continue
        
        is_long = long_entry.iloc[ei]
        entry = df['close'].iloc[ei]
        
        if is_long:
            sl = nsl(ei) * 0.998
            tp = 99999  # LONG reversal — no fixed TP
            # Exit on SHORT signal
        else:
            sl = nsh(ei) * 1.002
            tp = entry - df['atr'].iloc[ei] * 3
        
        max_hold = 192  # 48h
        end = min(ei + max_hold, len(df))
        result = None
        exit_px = entry
        exi = ei
        
        for j in range(ei + 1, end):
            if is_long:
                if df['low'].iloc[j] <= sl:
                    result = 'SL'; exit_px = sl; exi = j; break
                # LONG reversal: exit on SHORT signal
                short_sig = (short_entry.iloc[j] and df['w_strength'].iloc[j] > ws)
                if short_sig:
                    result = 'REV'; exit_px = df['close'].iloc[j]; exi = j; break
            else:
                if df['high'].iloc[j] >= sl:
                    result = 'SL'; exit_px = sl; exi = j; break
                if df['low'].iloc[j] <= tp:
                    result = 'TP'; exit_px = tp; exi = j; break
        
        if result is None:
            result = 'TIME'; exit_px = df['close'].iloc[end-1]; exi = end-1
        
        pnl = (exit_px - entry) / entry * 100
        if is_long: pnl -= 0.2
        else: pnl = -pnl - 0.2
        
        trades.append({'is_long': is_long, 'result': result, 'pnl': pnl,
                       'entry_idx': ei, 'exit_idx': exi})
        in_trade = True
        exit_idx_done = exi
        equity += CAPITAL * (pnl / 100)
    
    n = len(trades)
    if n == 0:
        results.append({**cfg, 'trades': 0, 'portfolio': 1000, 'wr': 0})
        continue
    
    wins = [t for t in trades if t['pnl'] > 0]
    wr = len(wins) / n * 100
    
    # Sharpe
    pnls = [t['pnl'] for t in trades]
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0
    
    # Max DD
    eqs = [CAPITAL]
    for t in trades: eqs.append(eqs[-1] + CAPITAL * (t['pnl'] / 100))
    peak = np.maximum.accumulate(eqs)
    dd = (np.array(eqs) - peak) / peak * 100
    max_dd = dd.min()
    
    lt = [t for t in trades if t['is_long']]
    st = [t for t in trades if not t['is_long']]
    lwr = len([t for t in lt if t['pnl'] > 0]) / len(lt) * 100 if lt else 0
    swr = len([t for t in st if t['pnl'] > 0]) / len(st) * 100 if st else 0
    
    results.append({**cfg, 'trades': n, 'portfolio': equity, 'wr': wr,
                    'sharpe': sharpe, 'max_dd': max_dd,
                    'long_wr': lwr, 'short_wr': swr,
                    'long_n': len(lt), 'short_n': len(st),
                    'rev_count': sum(1 for t in trades if t['result']=='REV'),
                    'tp_count': sum(1 for t in trades if t['result']=='TP'),
                    'sl_count': sum(1 for t in trades if t['result']=='SL')})
    
    print(f"  [{ci+1}/{len(configs)}] {cfg['name']}: {n}T | WR:{wr:.0f}% | ${equity:,.0f} | L/S:{lwr:.0f}/{swr:.0f} | R:{sum(1 for t in trades if t['result']=='REV')}", flush=True)

# ─── Print results ──────────────────────────────────────────────
rd = pd.DataFrame(results).sort_values('portfolio', ascending=False)

print(f"\n{'='*70}")
print(f"🐋 WHALE v3: 100-bar + PMA200 + LONG Reversal")
print(f"{'='*70}")
print(f"\n🏆 RANKED:\n")
print(f"{'#':<3} {'Config':<22} {'Trades':>6} {'PF':>8} {'WR':>5} {'L/S_WR':>10} {'Sharpe':>6} {'DD%':>6} {'Rev/TP/SL':>12}")
print("-"*85)

for rank, (_, r) in enumerate(rd.iterrows(), 1):
    if rank > 12: break
    ls = f"{r['long_wr']:.0f}/{r['short_wr']:.0f}"
    rev = f"{r['rev_count']}/{r['tp_count']}/{r['sl_count']}"
    print(f"{rank:<3} {r['name']:<22} {r['trades']:>6} ${r['portfolio']:>7,.0f} {r['wr']:>4.0f}% {ls:>10} {r['sharpe']:>5.2f} {r['max_dd']:>5.1f}% {rev:>12}")
