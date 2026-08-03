#!/usr/bin/env python3
"""
5-Strategy Shootout — FET/USDT — All Timeframes
Strategies ported from Pine Script to Python
Close-only simulation
"""

import ccxt, pandas as pd, numpy as np, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, '/data/trading28')
os.environ['PYTHONUNBUFFERED'] = '1'

SYMBOL = 'FET/USDT'
COMM = 0.002  # 0.2%
DAYS = 180
CAPITAL = 1000
TIMEFRAMES = ['3m', '5m', '15m', '30m', '1h', '4h']

# ═══════════ HELPERS ═══════════

def fetch_ohlcv(symbol, tf, days):
    exchange = ccxt.binance({'timeout': 15000})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_candles = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch: break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_supertrend(df, factor, period):
    atr = (df['high'] - df['low']).rolling(period).mean()
    hl2 = (df['high'] + df['low']) / 2
    up = hl2 - factor * atr
    dn = hl2 + factor * atr
    trend = pd.Series(0, index=df.index, dtype=int)
    up_line = pd.Series(np.nan, index=df.index)
    dn_line = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        pc = df['close'].iloc[i-1]
        if pd.notna(dn_line.iloc[i-1]) and pc < dn_line.iloc[i-1]:
            trend.iloc[i] = -1
            dn_line.iloc[i] = min(dn.iloc[i], dn_line.iloc[i-1])
        elif pd.notna(up_line.iloc[i-1]) and pc > up_line.iloc[i-1]:
            trend.iloc[i] = 1
            up_line.iloc[i] = max(up.iloc[i], up_line.iloc[i-1])
        elif pc > dn_line.iloc[i-1] if pd.notna(dn_line.iloc[i-1]) else False:
            trend.iloc[i] = 1
            up_line.iloc[i] = up.iloc[i]
        else:
            trend.iloc[i] = -1
            dn_line.iloc[i] = dn.iloc[i]
    return trend

def compute_metrics(trades_df, equity_curve, days):
    if len(trades_df) == 0:
        return {'n_trades': 0, 'wr': 0, 'net_pnl': 0, 'equity': CAPITAL, 'dd': 0, 'sharpe': 0, 'annual': 0}
    n = len(trades_df)
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]
    nw, nl = len(wins), len(losses)
    wr = nw / n * 100
    avg_win = wins['pnl_pct'].mean() if nw > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if nl > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    total_profit = wins['pnl_pct'].sum() if nw > 0 else 0
    total_loss = losses['pnl_pct'].sum() if nl > 0 else 0
    net_pnl = total_profit + total_loss
    final_equity = equity_curve[-1]
    eq = pd.Series(equity_curve)
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    daily_rets = eq.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 0 else 0
    annual = (final_equity / CAPITAL) ** (365 / days) - 1
    return {
        'n_trades': n, 'n_wins': nw, 'n_losses': nl, 'wr': wr,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'total_profit': total_profit, 'total_loss': total_loss, 'net_pnl': net_pnl,
        'equity': final_equity, 'dd': max_dd, 'sharpe': sharpe, 'annual': annual * 100,
    }

def always_in_reversal(idx, closes, long_signal, short_signal, warmup):
    """Always-in-market: flip position on opposite signal"""
    trades = []
    equity = CAPITAL
    equity_curve = [CAPITAL]
    pos = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    for i in range(warmup, len(closes)):
        if pos == 0:
            if long_signal[i]:
                pos, entry_price = 1, closes[i]
            elif short_signal[i]:
                pos, entry_price = -1, closes[i]
        elif pos == 1:
            if short_signal[i]:
                exit_px = closes[i]
                pnl = (exit_px / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos, entry_price = -1, closes[i]
        elif pos == -1:
            if long_signal[i]:
                exit_px = closes[i]
                pnl = (1 - exit_px / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos, entry_price = 1, closes[i]
        equity_curve.append(equity)
    if pos != 0:
        exit_px = closes[-1]
        pnl = ((exit_px / entry_price - 1) * 100 - COMM * 100) if pos == 1 else ((1 - exit_px / entry_price) * 100 - COMM * 100)
        trades.append({'pnl_pct': pnl})
        equity *= (1 + pnl / 100)
        equity_curve.append(equity)
    return trades, equity_curve

def fixed_tp_sl(idx, closes, highs, lows, long_signal, short_signal, warmup, tp_pct, sl_pct):
    """Fixed TP/SL with close-only exits"""
    trades = []
    equity = CAPITAL
    equity_curve = [CAPITAL]
    pos = 0
    entry_price = 0
    entry_i = 0
    for i in range(warmup, len(closes)):
        if pos == 0:
            if long_signal[i]:
                pos, entry_price, entry_i = 1, closes[i], i
            elif short_signal[i]:
                pos, entry_price, entry_i = -1, closes[i], i
        elif pos == 1:
            # Check TP
            if highs[i] >= entry_price * (1 + tp_pct/100):
                exit_px = entry_price * (1 + tp_pct/100)
                pnl = (exit_px / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos = 0
            # Check SL (close-only)
            elif closes[i] <= entry_price * (1 - sl_pct/100):
                exit_px = closes[i]
                pnl = (exit_px / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos = 0
            # Flip to short
            elif short_signal[i]:
                exit_px = closes[i]
                pnl = (exit_px / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos, entry_price = -1, closes[i]
        elif pos == -1:
            if lows[i] <= entry_price * (1 - tp_pct/100):
                exit_px = entry_price * (1 - tp_pct/100)
                pnl = (1 - exit_px / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos = 0
            elif closes[i] >= entry_price * (1 + sl_pct/100):
                exit_px = closes[i]
                pnl = (1 - exit_px / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos = 0
            elif long_signal[i]:
                exit_px = closes[i]
                pnl = (1 - exit_px / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl / 100)
                pos, entry_price = 1, closes[i]
        equity_curve.append(equity)
    if pos != 0:
        exit_px = closes[-1]
        pnl = ((exit_px / entry_price - 1) * 100 - COMM * 100) if pos == 1 else ((1 - exit_px / entry_price) * 100 - COMM * 100)
        trades.append({'pnl_pct': pnl})
        equity *= (1 + pnl / 100)
        equity_curve.append(equity)
    return trades, equity_curve

# ═══════════ STRATEGIES ═══════════

def strat1_bollinger_rsi(df):
    """Strategy 1: Bollinger + RSI Double Strategy (ChartArt)"""
    warmup = 200
    bb_basis = df['close'].rolling(200).mean()
    bb_std = df['close'].rolling(200).std()
    bb_upper = bb_basis + 2 * bb_std
    bb_lower = bb_basis - 2 * bb_std
    rsi_val = rsi(df['close'], 6)
    
    # RSI crosses over 50 AND price crosses over lower BB
    rsi_cross_up = (rsi_val > 50) & (rsi_val.shift(1) <= 50)
    price_cross_up = (df['close'] > bb_lower) & (df['close'].shift(1) <= bb_lower)
    long_sig = rsi_cross_up & price_cross_up
    
    # RSI crosses under 50 AND price crosses under upper BB
    rsi_cross_down = (rsi_val < 50) & (rsi_val.shift(1) >= 50)
    price_cross_down = (df['close'] < bb_upper) & (df['close'].shift(1) >= bb_upper)
    short_sig = rsi_cross_down & price_cross_down
    
    close_arr = df['close'].values
    long_arr = long_sig.values
    short_arr = short_sig.values
    trades, eq = always_in_reversal(df.index, close_arr, long_arr, short_arr, warmup)
    return trades, eq

def strat2_macd_sma200(df):
    """Strategy 2: MACD + SMA 200 (ChartArt)"""
    warmup = 200
    fast_ma = df['close'].rolling(12).mean()
    slow_ma = df['close'].rolling(26).mean()
    very_slow = df['close'].rolling(200).mean()
    macd = fast_ma - slow_ma
    signal = macd.rolling(9).mean()
    hist = macd - signal
    
    hist_cross_up = (hist > 0) & (hist.shift(1) <= 0)
    macd_pos = macd > 0
    fast_above_slow = fast_ma > slow_ma
    above_sma200 = df['close'].shift(26) > very_slow
    
    long_sig = hist_cross_up & macd_pos & fast_above_slow & above_sma200
    
    hist_cross_down = (hist < 0) & (hist.shift(1) >= 0)
    macd_neg = macd < 0
    fast_below_slow = fast_ma < slow_ma
    below_sma200 = df['close'].shift(26) < very_slow
    
    short_sig = hist_cross_down & macd_neg & fast_below_slow & below_sma200
    
    close_arr = df['close'].values
    long_arr = long_sig.values
    short_arr = short_sig.values
    trades, eq = always_in_reversal(df.index, close_arr, long_arr, short_arr, warmup)
    return trades, eq

def strat3_supertrend(df):
    """Strategy 3: SuperTrend (KivancOzbilgic)"""
    warmup = 50
    st = compute_supertrend(df, 3.0, 10)
    trend_up = st == 1
    trend_down = st == -1
    long_sig = trend_up & (st.shift(1) == -1)
    short_sig = trend_down & (st.shift(1) == 1)
    
    close_arr = df['close'].values
    long_arr = long_sig.values
    short_arr = short_sig.values
    trades, eq = always_in_reversal(df.index, close_arr, long_arr, short_arr, warmup)
    return trades, eq

def strat4_pmax(df):
    """Strategy 4: PMax Explorer — EMA(10) crossover with PMax"""
    warmup = 50
    ma = df['close'].ewm(span=10, adjust=False).mean()
    atr = (df['high'] - df['low']).rolling(10).mean()
    mult = 3.0
    up = ma - mult * atr
    dn = ma + mult * atr
    
    # PMax logic (simplified Supertrend on MA)
    trend = pd.Series(0, index=df.index, dtype=int)
    up_line = pd.Series(np.nan, index=df.index)
    dn_line = pd.Series(np.nan, index=df.index)
    for i in range(50, len(df)):
        prev_ma = ma.iloc[i-1]
        if pd.notna(dn_line.iloc[i-1]) and prev_ma < dn_line.iloc[i-1]:
            trend.iloc[i] = -1
            dn_line.iloc[i] = min(dn.iloc[i], dn_line.iloc[i-1])
        elif pd.notna(up_line.iloc[i-1]) and prev_ma > up_line.iloc[i-1]:
            trend.iloc[i] = 1
            up_line.iloc[i] = max(up.iloc[i], up_line.iloc[i-1])
        elif prev_ma > dn_line.iloc[i-1] if pd.notna(dn_line.iloc[i-1]) else False:
            trend.iloc[i] = 1
            up_line.iloc[i] = up.iloc[i]
        else:
            trend.iloc[i] = -1
            dn_line.iloc[i] = dn.iloc[i]
    
    long_sig = (trend == 1) & (trend.shift(1) == -1)
    short_sig = (trend == -1) & (trend.shift(1) == 1)
    
    close_arr = df['close'].values
    long_arr = long_sig.values
    short_arr = short_sig.values
    trades, eq = always_in_reversal(df.index, close_arr, long_arr, short_arr, warmup)
    return trades, eq

def strat5_3commas(df):
    """Strategy 5: 3Commas Bot — MA(21,50) crossover + ATR-based SL/TP"""
    warmup = 200
    ema1 = df['close'].ewm(span=21, adjust=False).mean()
    ema2 = df['close'].ewm(span=50, adjust=False).mean()
    atr = (df['high'] - df['low']).rolling(14).mean()
    lowest_low = df['low'].rolling(5).min()
    highest_high = df['high'].rolling(5).max()
    
    cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
    
    # SL: swing ± ATR, TP: R:R=1 × risk distance
    long_sl = lowest_low - atr  # RiskM=1
    short_sl = highest_high + atr
    long_risk = df['close'] - long_sl  # distance to SL
    short_risk = short_sl - df['close']
    long_tp_pct = (long_risk / df['close'] * 100)  # R:R=1 → TP% = risk%
    short_tp_pct = (short_risk / df['close'] * 100)
    
    long_sig = cross_up
    short_sig = cross_down
    
    # For this strategy, use fixed TP/SL on each entry
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    long_arr = long_sig.values
    short_arr = short_sig.values
    
    # Pre-compute TP/SL percentages per bar
    long_tp_arr = long_tp_pct.values
    short_tp_arr = short_tp_pct.values
    
    trades = []
    equity = CAPITAL
    equity_curve = [CAPITAL]
    pos = 0
    entry_price = 0
    entry_tp = 0
    entry_sl = 0
    for i in range(warmup, len(close_arr)):
        if pos == 0:
            if long_arr[i]:
                pos = 1
                entry_price = close_arr[i]
                tp_pct = max(long_tp_arr[i], 0.5)  # minimum 0.5% TP
                entry_tp = entry_price * (1 + tp_pct/100)
                entry_sl = entry_price * (1 - tp_pct/100)  # R:R=1
            elif short_arr[i]:
                pos = -1
                entry_price = close_arr[i]
                tp_pct = max(short_tp_arr[i], 0.5)
                entry_tp = entry_price * (1 - tp_pct/100)
                entry_sl = entry_price * (1 + tp_pct/100)
        elif pos == 1:
            if high_arr[i] >= entry_tp:
                pnl = (entry_tp / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = 0
            elif close_arr[i] <= entry_sl:
                pnl = (close_arr[i] / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = 0
            elif short_arr[i]:
                # Flip
                pnl = (close_arr[i] / entry_price - 1) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = -1
                entry_price = close_arr[i]
                tp_pct = max(short_tp_arr[i], 0.5)
                entry_tp = entry_price * (1 - tp_pct/100)
                entry_sl = entry_price * (1 + tp_pct/100)
        elif pos == -1:
            if low_arr[i] <= entry_tp:
                pnl = (1 - entry_tp / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = 0
            elif close_arr[i] >= entry_sl:
                pnl = (1 - close_arr[i] / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = 0
            elif long_arr[i]:
                pnl = (1 - close_arr[i] / entry_price) * 100 - COMM * 100
                trades.append({'pnl_pct': pnl})
                equity *= (1 + pnl/100)
                pos = 1
                entry_price = close_arr[i]
                tp_pct = max(long_tp_arr[i], 0.5)
                entry_tp = entry_price * (1 + tp_pct/100)
                entry_sl = entry_price * (1 - tp_pct/100)
        equity_curve.append(equity)
    
    if pos != 0:
        exit_px = close_arr[-1]
        pnl = ((exit_px / entry_price - 1) * 100 - COMM * 100) if pos == 1 else ((1 - exit_px / entry_price) * 100 - COMM * 100)
        trades.append({'pnl_pct': pnl})
        equity *= (1 + pnl/100)
        equity_curve.append(equity)
    
    return trades, equity_curve

# ═══════════ MAIN ═══════════

STRATEGIES = [
    ("1-BB+RSI", strat1_bollinger_rsi),
    ("2-MACD+SMA200", strat2_macd_sma200),
    ("3-SuperTrend", strat3_supertrend),
    ("4-PMax Explorer", strat4_pmax),
    ("5-3Commas Bot", strat5_3commas),
]

print("Fetching FET/USDT all timeframes...")
data = {}
for tf in TIMEFRAMES:
    data[tf] = fetch_ohlcv(SYMBOL, tf, DAYS)
    print(f"  {tf}: {len(data[tf])} candles")

print(f"\n{'='*90}")
print(f"5-STRATEGY SHOOTOUT — FET/USDT — {DAYS} days — All Timeframes")
print(f"{'='*90}")

all_results = []

for strat_name, strat_fn in STRATEGIES:
    print(f"\n─── {strat_name} ───")
    for tf in TIMEFRAMES:
        df = data[tf].copy()
        trades, eq = strat_fn(df)
        m = compute_metrics(pd.DataFrame(trades) if trades else pd.DataFrame(columns=['pnl_pct']), eq, DAYS)
        if m['n_trades'] == 0:
            print(f"  {tf:>4}: 0 trades")
            continue
        all_results.append({
            'strategy': strat_name,
            'tf': tf,
            **m
        })
        print(f"  {tf:>4}: {m['n_trades']:>4d} trades | WR {m['wr']:>5.1f}% | R:R {m['rr']:>5.2f}x | DD {m['dd']:>6.1f}% | ${m['equity']-1000:>+8.0f} | Sharpe {m['sharpe']:>5.2f}")

# ═══════════ RANKING ═══════════
print(f"\n{'='*90}")
print("🏆 RANKING — by Portfolio Return")
print(f"{'='*90}")
by_return = sorted(all_results, key=lambda x: x['equity'], reverse=True)
for i, r in enumerate(by_return[:15]):
    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
    print(f"{icon:>3} {r['strategy']:>18} {r['tf']:>4} | {r['n_trades']:>4d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | ${r['equity']-1000:>+8.0f} | {r['annual']:>+6.1f}% yr | Sharpe {r['sharpe']:>5.2f}")

print(f"\n{'='*90}")
print("🏆 RANKING — by Win Rate")
print(f"{'='*90}")
by_wr = sorted(all_results, key=lambda x: x['wr'], reverse=True)
for i, r in enumerate(by_wr[:15]):
    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
    print(f"{icon:>3} {r['strategy']:>18} {r['tf']:>4} | {r['n_trades']:>4d}t | WR {r['wr']:>5.1f}% | DD {r['dd']:>6.1f}% | ${r['equity']-1000:>+8.0f} | Sharpe {r['sharpe']:>5.2f}")
