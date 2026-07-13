#!/usr/bin/env python3
"""Run SMA Crossover backtest and print results."""
import ccxt
import pandas as pd
import numpy as np
import sys
from ta.trend import SMAIndicator

# --- Backtest Engine (compact) ---
def backtest(df, capital=10000, fee=0.001):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['position'] = df['signal'].shift(1).fillna(0)
    trade_signal = df['signal'].diff().fillna(0) != 0
    df['cost'] = 0.0
    df.loc[trade_signal, 'cost'] = fee
    df['strat_ret'] = df['position'] * df['returns'] - df['cost']
    df['equity'] = (1 + df['strat_ret'].fillna(0)).cumprod() * capital
    df['bench'] = (1 + df['returns'].fillna(0)).cumprod() * capital

    total_ret = (df['equity'].iloc[-1] / capital - 1) * 100
    bench_ret = (df['bench'].iloc[-1] / capital - 1) * 100
    mean_r = df['strat_ret'].mean()
    std_r = df['strat_ret'].std()
    sharpe = (mean_r / std_r * np.sqrt(365*24)) if std_r > 0 else 0

    peak = df['equity'].expanding().max()
    dd = (df['equity'] - peak) / peak * 100
    max_dd = dd.min()

    trades = df['strat_ret'][df['strat_ret'] != 0]
    win_rate = (trades > 0).sum() / len(trades) * 100 if len(trades) > 0 else 0

    n_trades = (df['signal'].diff().fillna(0) != 0).sum()

    return total_ret, bench_ret, sharpe, max_dd, win_rate, n_trades

# --- Main ---
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
TF = sys.argv[2] if len(sys.argv) > 2 else '1h'
FAST = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SLOW = int(sys.argv[4]) if len(sys.argv) > 4 else 50

print(f"🔍 SMA Crossover Backtest: {SYMBOL} {TF}")
print(f"   Fast SMA: {FAST} | Slow SMA: {SLOW}")
print(f"   Fetching data...")

exchange = ccxt.binance()
all_candles = []

# Fetch up to 90 days (need multiple calls for >1000 candles)
since = exchange.parse8601('2026-04-01T00:00:00Z')
while True:
    candles = exchange.fetch_ohlcv(SYMBOL, TF, since=since, limit=1000)
    if not candles:
        break
    all_candles.extend(candles)
    since = candles[-1][0] + 1
    if len(candles) < 1000:
        break

df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True)

print(f"   Loaded {len(df)} candles ({df.index[0]} → {df.index[-1]})")

# Generate signals
df['sma_fast'] = SMAIndicator(df['close'], window=FAST).sma_indicator()
df['sma_slow'] = SMAIndicator(df['close'], window=SLOW).sma_indicator()
df['signal'] = 0
df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1

# Run backtest
total_ret, bench_ret, sharpe, max_dd, win_rate, n_trades = backtest(df)

print(f"\n📊 Results:")
print(f"   Strategy Return:  {total_ret:+.2f}%")
print(f"   Buy & Hold:       {bench_ret:+.2f}%")
print(f"   Sharpe Ratio:     {sharpe:.2f}")
print(f"   Max Drawdown:     {max_dd:.2f}%")
print(f"   Win Rate:         {win_rate:.1f}%")
print(f"   Total Trades:     {n_trades}")

# Current signal
last_signal = df['signal'].iloc[-1]
signal_text = "🟢 BULLISH (Long)" if last_signal == 1 else "🔴 BEARISH (Short/Cash)"
print(f"\n📍 Current Signal: {signal_text}")
print(f"   Price: ${df['close'].iloc[-1]:,.2f}")
print(f"   SMA({FAST}): ${df['sma_fast'].iloc[-1]:,.2f}")
print(f"   SMA({SLOW}): ${df['sma_slow'].iloc[-1]:,.2f}")
