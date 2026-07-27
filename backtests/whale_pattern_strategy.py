#!/usr/bin/env python3
"""
NEW STRATEGY: Whale Pattern Strategy (1h timeframe)
Based on 30-day pattern discovery — 100% of top gainers had whale bars.
Entry: whale bar on 1h + optional filters
Exit: TP / SL based on avg pump stats
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────
DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)

BACKTEST_DAYS = 60          # how far back to test
TIMEFRAME = '1h'            # hourly candles
LOOKBACK_CANDLES = 500      # max candles to fetch

# Strategy parameters (based on pattern discovery)
WHALE_STD_THRESHOLD = 2.0   # volume > mean + 2*std = whale
MIN_WHALE_BARS = 1          # at least 1 whale bar in lookback window
RSI_MAX = 50                # optional: only enter if RSI < 50
REQUIRE_PREV_DAY_RED = False # optional: require previous daily candle red
REQUIRE_PRICE_BOTTOM = False  # optional: price in bottom 30% of range

TP_PCT = 0.07               # +7% take profit (close to median +7.3%)
SL_PCT = 0.03               # -3% stop loss
MAX_HOLD_HOURS = 48         # max hold time
COMMISSION = 0.002          # 0.2%

INITIAL_CAPITAL = 1000

# ── Load coins ──────────────────────────────────────────
with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
halal_coins = config['halal'] + config['halal2']
seen = set()
halal_coins = [c for c in halal_coins if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
halal_coins = [c for c in halal_coins if c not in blacklist]

print(f"🎯 Whale Pattern Strategy — Backtest")
print(f"   Coins: {len(halal_coins)}")
print(f"   Period: {BACKTEST_DAYS} days")
print(f"   Entry: 1h whale bar (vol > {WHALE_STD_THRESHOLD}σ)")
print(f"   TP: +{TP_PCT*100:.0f}% | SL: -{SL_PCT*100:.0f}%")
print(f"   Capital: ${INITIAL_CAPITAL}")

# ── Exchange singleton ──────────────────────────────────
_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

# ── Step 1: Fetch 1h data for all coins ─────────────────
def fetch_hourly_data(coins):
    """Fetch 1h OHLCV for all coins."""
    cache_file = os.path.join(DATA_DIR, 'hourly_60d.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        print(f"📦 Loaded hourly cache: {len(data)} coins")
        return data
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=BACKTEST_DAYS)).isoformat())
    
    all_data = {}
    errors = 0
    
    for i, coin in enumerate(coins):
        symbol = f"{coin}/USDT"
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=LOOKBACK_CANDLES)
            if len(ohlcv) >= 100:
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                all_data[coin] = {
                    'ts': [int(o[0]) for o in ohlcv],
                    'open': [float(o[1]) for o in ohlcv],
                    'high': [float(o[2]) for o in ohlcv],
                    'low': [float(o[3]) for o in ohlcv],
                    'close': [float(o[4]) for o in ohlcv],
                    'volume': [float(o[5]) for o in ohlcv],
                }
            if (i+1) % 20 == 0:
                print(f"  📊 {i+1}/{len(coins)} coins...")
        except Exception as e:
            errors += 1
        time.sleep(0.05)
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ Fetched {len(all_data)} coins ({errors} errors)")
    return all_data

# ── Step 2: Run backtest ────────────────────────────────
def run_backtest(all_data):
    """Backtest whale pattern strategy on all coins."""
    
    all_trades = []
    equity_curve = []
    capital = INITIAL_CAPITAL
    active_positions = []  # list of {coin, entry_price, entry_time, tp, sl, size}
    
    # Process all coins and collect all whale signals with timestamps
    all_signals = []  # (coin, timestamp_idx, price, rsi, whale_count, ...)
    
    for coin, data in all_data.items():
        close = np.array(data['close'])
        volume = np.array(data['volume'])
        ts = data['ts']
        
        if len(close) < 50:
            continue
        
        # Rolling mean and std for whale detection
        vol_mean = pd.Series(volume).rolling(24).mean()
        vol_std = pd.Series(volume).rolling(24).std()
        
        # RSI(14)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Price position (last 48h range)
        price_pos = pd.Series(0.5, index=range(len(close)))
        for i in range(48, len(close)):
            range_hi = max(close[i-48:i])
            range_lo = min(close[i-48:i])
            if range_hi > range_lo:
                price_pos.iloc[i] = (close[i] - range_lo) / (range_hi - range_lo)
        
        # Detect whale bars
        for i in range(48, len(close) - 1):  # -1 because we enter NEXT candle
            vol_threshold = vol_mean.iloc[i] + WHALE_STD_THRESHOLD * vol_std.iloc[i]
            
            if volume[i] > vol_threshold and vol_std.iloc[i] > 0:
                # Whale detected!
                entry_rsi = rsi.iloc[i]
                entry_pos = price_pos.iloc[i]
                
                # Apply filters
                if entry_rsi > RSI_MAX:
                    continue
                
                # Store signal
                all_signals.append({
                    'coin': coin,
                    'idx': i + 1,  # enter on NEXT candle
                    'entry_price': close[i + 1] if i + 1 < len(close) else close[i],
                    'ts': ts[i + 1] if i + 1 < len(ts) else ts[i],
                    'rsi': float(entry_rsi) if not pd.isna(entry_rsi) else 50,
                    'price_pos': float(entry_pos) if not pd.isna(entry_pos) else 0.5,
                })
    
    print(f"\n🎯 Signals found: {len(all_signals)}")
    
    # Sort signals by timestamp
    all_signals.sort(key=lambda s: s['ts'])
    
    # Process trades in chronological order
    for sig in all_signals:
        coin = sig['coin']
        entry_price = sig['entry_price']
        entry_ts = sig['ts']
        
        # Check if coin already in active position
        if any(p['coin'] == coin for p in active_positions):
            continue
        
        # Get remaining candles for exit simulation
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        ts_arr = data['ts']
        
        # Find entry index in the coin's data
        try:
            entry_idx = ts_arr.index(entry_ts)
        except ValueError:
            continue
        
        # Position size
        position_size = capital * 0.10  # 10% per trade
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        
        # Simulate exit
        exit_price = None
        exit_type = None
        exit_idx = None
        
        for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD_HOURS, len(close_arr))):
            if low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_type = 'SL'
                exit_idx = j
                break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price
                exit_type = 'TP'
                exit_idx = j
                break
        
        if exit_price is None and entry_idx + MAX_HOLD_HOURS < len(close_arr):
            exit_price = close_arr[entry_idx + MAX_HOLD_HOURS]
            exit_type = 'TIME'
            exit_idx = entry_idx + MAX_HOLD_HOURS
        elif exit_price is None:
            exit_price = close_arr[-1]
            exit_type = 'EOD'
            exit_idx = len(close_arr) - 1
        
        # Calculate P&L
        gross_pnl_pct = (exit_price / entry_price - 1) * 100
        net_pnl_pct = gross_pnl_pct - COMMISSION * 100
        net_pnl_usd = position_size * net_pnl_pct / 100
        
        # Update capital
        capital += net_pnl_usd
        
        trade_record = {
            'coin': coin,
            'entry_price': round(entry_price, 8),
            'exit_price': round(exit_price, 8),
            'entry_time': datetime.fromtimestamp(entry_ts/1000).strftime('%Y-%m-%d %H:%M'),
            'exit_time': datetime.fromtimestamp(ts_arr[exit_idx]/1000).strftime('%Y-%m-%d %H:%M') if exit_idx else '?',
            'exit_type': exit_type,
            'pnl_pct': round(net_pnl_pct, 2),
            'pnl_usd': round(net_pnl_usd, 2),
            'rsi': sig['rsi'],
            'capital_after': round(capital, 2),
            'hold_hours': exit_idx - entry_idx if exit_idx else 0,
        }
        all_trades.append(trade_record)
        equity_curve.append(capital)
        
        # Track active positions (for preventing duplicates in same timeframe)
        active_positions.append({
            'coin': coin,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx if exit_idx else 99999,
        })
        
        # Clean old positions
        active_positions = [p for p in active_positions if p['exit_idx'] > entry_idx]
    
    return all_trades, equity_curve, capital

# ── Step 3: Analyze results ─────────────────────────────
def analyze_results(trades, equity_curve, final_capital):
    """Print detailed backtest results."""
    
    if not trades:
        print("\n❌ No trades found!")
        return
    
    df = pd.DataFrame(trades)
    
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100
    
    # Calculate equity curve stats
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"   Total Trades:    {len(df)}")
    print(f"   Win Rate:        {wr:.1f}%")
    print(f"   Wins:            {len(wins)} | Losses: {len(losses)}")
    print(f"   Avg Win:         +{wins['pnl_pct'].mean():.2f}%" if len(wins) > 0 else "")
    print(f"   Avg Loss:        {losses['pnl_pct'].mean():.2f}%" if len(losses) > 0 else "")
    print(f"   Avg Trade:       {df['pnl_pct'].mean():.2f}%")
    print(f"   Profit Factor:   {abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()):.2f}" if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else "")
    print(f"   Total Return:    {total_return:+.1f}%")
    print(f"   Final Capital:   ${final_capital:,.2f}")
    print(f"   Max Drawdown:    {max_dd:.2f}%")
    print(f"   Avg Hold:        {df['hold_hours'].mean():.1f}h")
    
    # By exit type
    print(f"\n## By Exit Type:")
    for etype in ['TP', 'SL', 'TIME', 'EOD']:
        subset = df[df['exit_type'] == etype]
        if len(subset) > 0:
            print(f"   {etype:6s}: {len(subset)} trades, avg {subset['pnl_pct'].mean():+.2f}%")
    
    # Top coins
    print(f"\n## Top Performing Coins:")
    coin_stats = df.groupby('coin').agg(
        trades=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_usd', 'sum')
    ).sort_values('total_pnl', ascending=False)
    for coin, row in coin_stats.head(10).iterrows():
        print(f"   {coin:8s}: {int(row['trades'])} trades, avg {row['avg_pnl']:+.2f}%, total ${row['total_pnl']:+.2f}")
    
    # RSI vs performance
    print(f"\n## RSI at Entry vs Performance:")
    for rsi_range, (lo, hi) in [('<30', (0,30)), ('30-40', (30,40)), ('40-50', (40,50))]:
        subset = df[(df['rsi'] >= lo) & (df['rsi'] < hi)]
        if len(subset) > 0:
            sub_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            print(f"   RSI {rsi_range}: {len(subset)} trades, WR {sub_wr:.0f}%, avg {subset['pnl_pct'].mean():+.2f}%")
    
    # Save report
    report = {
        'params': {
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT,
            'whale_std': WHALE_STD_THRESHOLD,
            'rsi_max': RSI_MAX,
            'max_hold_hours': MAX_HOLD_HOURS,
        },
        'summary': {
            'total_trades': len(df),
            'win_rate': round(wr, 1),
            'total_return_pct': round(total_return, 1),
            'final_capital': round(final_capital, 2),
            'max_dd_pct': round(max_dd, 2),
            'profit_factor': round(abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()), 2) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else None,
        },
        'trades': [{
            'coin': t['coin'],
            'entry': t['entry_price'],
            'exit': t['exit_price'],
            'entry_time': t['entry_time'],
            'exit_time': t['exit_time'],
            'type': t['exit_type'],
            'pnl_pct': t['pnl_pct'],
        } for t in trades[-50:]],  # last 50 trades
    }
    
    report_path = os.path.join(DATA_DIR, 'strategy_backtest.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved: {report_path}")
    
    return df

# ── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🐋 WHALE PATTERN STRATEGY — 1h Backtest")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n── Fetching 1h data ──")
    all_data = fetch_hourly_data(halal_coins)
    
    # Step 2: Run backtest
    print("\n── Running backtest ──")
    trades, equity_curve, final_capital = run_backtest(all_data)
    
    # Step 3: Analyze
    df = analyze_results(trades, equity_curve, final_capital)
    
    print(f"\n✅ Done!")
    return df

if __name__ == '__main__':
    df = main()
