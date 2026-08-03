#!/usr/bin/env python3
"""
5-Lines Strategy Backtest — FET/USDT
Buy: price crosses above EMA9
Exit: price crosses below EMA21 OR Supertrend flip
"""
import ccxt, numpy as np, pandas as pd, json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COMMISSION = 0.002
INITIAL_CAPITAL = 1000
SYMBOL = 'FET/USDT'
DATA_DIR = '/data/trading28/data'
os.makedirs(DATA_DIR, exist_ok=True)

# ── Fetch data ──────────────────────────────────────────
def fetch_fet(tf, days):
    cache_key = f'fet_{tf}_{days}d'
    cache_file = os.path.join(DATA_DIR, f'{cache_key}.json')
    
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        print(f"📦 Cache: {cache_key} ({len(data['close'])} candles)")
        return data
    
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.now() - timedelta(days=days+3)).isoformat())
    ohlcv = exchange.fetch_ohlcv(SYMBOL, tf, since=since, limit=10000)
    
    data = {
        'ts': [int(o[0]) for o in ohlcv],
        'open': [float(o[1]) for o in ohlcv],
        'high': [float(o[2]) for o in ohlcv],
        'low': [float(o[3]) for o in ohlcv],
        'close': [float(o[4]) for o in ohlcv],
        'volume': [float(o[5]) for o in ohlcv],
    }
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    print(f"✅ Fetched: {cache_key} ({len(data['close'])} candles)")
    return data

# ── Supertrend ──────────────────────────────────────────
def calc_supertrend(c, h, l, period=10, factor=3.0):
    n = len(c)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    hl2 = (np.array(h) + np.array(l)) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    
    trend = np.ones(n)
    st_line = np.full(n, np.nan)
    
    for i in range(period, n):
        if c[i-1] > upper[i-1]:
            trend[i] = 1
        elif c[i-1] < lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            if trend[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        st_line[i] = lower[i] if trend[i] == 1 else upper[i]
    
    return st_line, trend

# ── EMAs ────────────────────────────────────────────────
def calc_ema(arr, span):
    return pd.Series(arr).ewm(span=span).mean().values

# ── Backtest ────────────────────────────────────────────
def backtest(data, tp=None, sl=None, max_hold_candles=None, use_st=False, use_trend_filter=False):
    c = np.array(data['close'])
    h = np.array(data['high'])
    l = np.array(data['low'])
    n = len(c)
    
    # Indicators
    ema9 = calc_ema(c, 9)
    ema21 = calc_ema(c, 21)
    ema50 = calc_ema(c, 50)
    st_line, st_trend = calc_supertrend(c, h, l, 10, 3)
    
    # Slope
    slope = np.full(n, np.nan)
    angle_bars = 5
    for i in range(angle_bars, n):
        slope[i] = (ema50[i] - ema50[i-angle_bars]) / ema50[i-angle_bars] * 100
    trend_up = slope > 0.08
    
    # Signals
    trades = []
    in_trade = False
    entry_idx = entry_px = 0
    
    for i in range(1, n):
        # Buy: close crosses above EMA9
        buy_signal = c[i] > ema9[i] and c[i-1] <= ema9[i-1]
        
        if buy_signal and not in_trade:
            # Trend filter
            if use_trend_filter and not trend_up[i]:
                continue
            in_trade = True
            entry_idx = i
            entry_px = c[i]
            continue
        
        if not in_trade:
            continue
        
        exit_px = exit_type = None
        exit_idx = i
        
        # Exit conditions
        sell_cross = c[i] < ema21[i] and c[i-1] >= ema21[i-1]
        st_flip = use_st and st_trend[i] != st_trend[i-1] and st_trend[i] != st_trend[entry_idx]
        time_exit = max_hold_candles and (i - entry_idx >= max_hold_candles)
        
        # TP/SL hit during bar
        if tp and h[i] >= entry_px * (1+tp):
            exit_px = entry_px * (1+tp)
            exit_type = 'TP'
        elif sl and l[i] <= entry_px * (1-sl):
            exit_px = entry_px * (1-sl)
            exit_type = 'SL'
        elif sell_cross:
            exit_px = c[i]
            exit_type = 'SELL'
        elif st_flip:
            exit_px = c[i]
            exit_type = 'STOP'
        elif time_exit:
            exit_px = c[i]
            exit_type = 'TIME'
        
        if exit_px is not None:
            pnl_pct = (exit_px / entry_px - 1) * 100 - COMMISSION * 100
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry': entry_px,
                'exit': exit_px,
                'pnl_pct': pnl_pct,
                'type': exit_type,
                'bars': exit_idx - entry_idx,
            })
            in_trade = False
    
    return trades

# ── Metrics ─────────────────────────────────────────────
def calc_metrics(trades, capital=INITIAL_CAPITAL, days=365):
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0, 'pnl_net': 0}
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    
    capital_curve = [capital]
    for t in trades:
        sz = capital_curve[-1] * 0.10
        capital_curve.append(capital_curve[-1] + sz * t['pnl_pct'] / 100)
    
    final = capital_curve[-1]
    net_pnl = (final / capital - 1) * 100
    
    # Sharpe
    daily_returns = []
    if len(capital_curve) > 1:
        daily_returns = np.diff(capital_curve) / capital_curve[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    # Max drawdown
    peak = np.maximum.accumulate(capital_curve)
    dd = (capital_curve - peak) / peak * 100
    max_dd = np.min(dd)
    
    # Avg win/loss
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    
    wr = len(wins) / len(trades) * 100
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # TP/SL/TIME/SELL counts
    tp_count = sum(1 for t in trades if t['type'] == 'TP')
    sl_count = sum(1 for t in trades if t['type'] == 'SL')
    time_count = sum(1 for t in trades if t['type'] == 'TIME')
    sell_count = sum(1 for t in trades if t['type'] == 'SELL')
    stop_count = sum(1 for t in trades if t['type'] == 'STOP')
    
    annual_return = ((final / capital) ** (365 / days) - 1) * 100 if days > 0 else 0
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pnl_net': net_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': rr,
        'annual': annual_return,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'time_count': time_count,
        'sell_count': sell_count,
        'stop_count': stop_count,
        'final_capital': final,
        'bars_avg': np.mean([t['bars'] for t in trades]),
    }

# ── Run ─────────────────────────────────────────────────
timeframes = {'15m': 96, '1h': 24, '4h': 6}  # max hold in candles

print("=" * 60)
print(f"🧪 5-Lines Strategy Backtest — {SYMBOL}")
print("=" * 60)

for tf, max_hold_c in timeframes.items():
    print(f"\n{'─'*60}")
    print(f"⏱️ {tf.upper()}")
    print(f"{'─'*60}")
    
    for days in [30, 90, 365]:
        data = fetch_fet(tf, days)
        
        # Strategy variants
        configs = [
            ("EMA9↗EMA21↘", False, None, None, None),
            ("+TP1.5%/SL1%", False, 0.015, 0.01, None),
            ("+Supertrend stop", True, None, None, None),
            ("+ST+TP/SL+Trend↑", True, 0.02, 0.015, None),
        ]
        
        for label, use_st, tp, sl, _ in configs:
            trades = backtest(data, tp=tp, sl=sl, max_hold_candles=max_hold_c*4,
                            use_st=use_st, use_trend_filter=('Trend' in label))
            m = calc_metrics(trades, days=days)
            
            # Format report line
            pnl_str = f"${m['final_capital']:.0f}"
            wr_str = f"{m['wr']:.1f}%"
            dd_str = f"{m['max_dd']:.1f}%"
            trades_n = m['trades']
            
            if trades_n > 0:
                print(f"  {label:<22s} | {days:>3d}d | {trades_n:>3d} trades | WR {wr_str:>6s} | "
                      f"💰 {pnl_str:>7s} | DD {dd_str:>6s} | Sharpe {m['sharpe']:.2f} | "
                      f"🎯{m['tp_count']} 🛑{m['sl_count']} 🐌{m['time_count']} "
                      f"📉{m['sell_count']} ⏹{m['stop_count']} | "
                      f"Avg W {m['avg_win']:+.2f}% L {m['avg_loss']:+.2f}%")
            else:
                print(f"  {label:<22s} | {days:>3d}d | 0 trades")

print("\n✅ Done")
